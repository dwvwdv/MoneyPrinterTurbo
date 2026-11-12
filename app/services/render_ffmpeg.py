"""FFmpeg render backend: each render step is one FFmpeg process and one encode.

The MoviePy backend decodes every material slice in Python, encodes it to a
temporary MP4, re-encodes all of them again in an FFmpeg concat, and then
decodes that result in Python once more to draw subtitles and encode the final
video. Frames cross the Python/FFmpeg pipe three times and are encoded three
times.

This backend keeps the same two entry points and contracts, but builds an
FFmpeg filter graph instead:

- ``combine_videos`` picks the same slices with the same allocation rules, then
  trims, retimes, fits and fades all of them inside one FFmpeg filter graph
  and encodes ``combined-N.mp4`` once.
- ``generate_video`` draws each subtitle with the shared MoviePy subtitle code
  into a transparent PNG (so text looks the same as the MoviePy backend), then
  overlays them and mixes narration and BGM in one FFmpeg pass.

Options that FFmpeg cannot reproduce yet (slide/zoom/shuffle transitions and
the subtitle spring animation) are detected before any work starts, and that
step is delegated to the MoviePy backend. Once FFmpeg has started, a failure is
raised as is: the partial output is removed and nothing is re-rendered.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Callable, List

import numpy as np
from loguru import logger
from PIL import Image

from app.config import config
from app.models.schema import (
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import bgm as bgm_service
from app.services import video
from app.services.utils import subtitle_shaping
from app.utils import file_security, logging_utils, utils
from app.utils.subtitle_reader import moviepy_subtitles

# Transitions this backend draws itself. The others are rendered by MoviePy.
_SUPPORTED_TRANSITIONS = {
    None,
    "",
    VideoTransitionMode.none.value,
    VideoTransitionMode.fade_in.value,
    VideoTransitionMode.fade_out.value,
}
_SPRING_ANIMATIONS = ("pop_spring", "spring", "pop")
_BGM_FADE_OUT_SECONDS = 3.0


_DEFAULT_COMBINE_CHUNK_SIZE = 6


def _get_combine_chunk_size() -> int:
    try:
        size = int(config.app.get("ffmpeg_combine_chunk_size", _DEFAULT_COMBINE_CHUNK_SIZE))
    except (TypeError, ValueError):
        size = _DEFAULT_COMBINE_CHUNK_SIZE
    return max(1, min(64, size))


class UnsupportedRenderOption(Exception):
    """Raised before rendering starts when FFmpeg cannot reproduce an option."""


# ---------------------------------------------------------------------------
# probing


def ffprobe_binary() -> str | None:
    """Return ffprobe next to the configured FFmpeg, or on PATH, or None.

    The imageio-ffmpeg fallback bundles ffmpeg without ffprobe, so a working
    ffmpeg does not imply that ffprobe exists.
    """
    ffmpeg_binary = utils.get_ffmpeg_binary()
    directory, name = os.path.split(ffmpeg_binary)
    if directory:
        candidate = os.path.join(directory, name.replace("ffmpeg", "ffprobe"))
        if candidate != ffmpeg_binary and os.path.isfile(candidate):
            return candidate
    return shutil.which("ffprobe")


def _ffprobe_binary() -> str:
    binary = ffprobe_binary()
    if binary is None:
        raise FileNotFoundError("ffprobe not found")
    return binary


def _probe(path: str) -> dict:
    result = subprocess.run(
        [
            _ffprobe_binary(), "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", path,
        ],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "ffprobe failed").strip())
    return json.loads(result.stdout or "{}")


def _media_duration(info: dict, stream_type: str | None = None) -> float:
    for stream in info.get("streams", []):
        if stream_type and stream.get("codec_type") != stream_type:
            continue
        try:
            duration = float(stream.get("duration"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            return duration
    return float(info.get("format", {}).get("duration") or 0.0)


def _video_geometry(info: dict) -> tuple[int, int, float]:
    """Return the displayed width, height and duration of the first video stream."""
    for stream in info.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        rotation = 0
        for side_data in stream.get("side_data_list", []) or []:
            if "rotation" in side_data:
                rotation = int(side_data["rotation"])
        rotation = int(stream.get("tags", {}).get("rotate", rotation) or 0)
        # FFmpeg auto-rotates on decode, as MoviePy does, so use the displayed size.
        if abs(rotation) % 180 == 90:
            width, height = height, width
        duration = _media_duration(info, "video")
        return width, height, duration
    raise ValueError("no video stream")


def _audio_sample_rate(info: dict) -> int:
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            try:
                return int(stream.get("sample_rate") or 0)
            except (TypeError, ValueError):
                return 0
    return 0


# ---------------------------------------------------------------------------
# running FFmpeg


def _run_ffmpeg(
    command: list[str],
    *,
    description: str,
    total_seconds: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> None:
    """Run FFmpeg, forward ``-progress`` output as seconds rendered, and raise on failure."""
    with tempfile.TemporaryFile(mode="w+b") as stderr_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
        stop_event = threading.Event()
        rendered = {"seconds": 0.0}

        def log_heartbeat() -> None:
            while not stop_event.wait(video._STAGE_HEARTBEAT_SECONDS):
                total = f" of {total_seconds:.0f}s" if total_seconds else ""
                logger.info(
                    f"{description} still running: "
                    f"{rendered['seconds']:.0f}s{total} rendered"
                )

        reporter = threading.Thread(
            target=logging_utils.bind_log_scope(log_heartbeat), daemon=True
        )
        reporter.start()
        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("out_time_us="):
                    continue
                try:
                    seconds = int(line.split("=", 1)[1]) / 1_000_000
                except ValueError:
                    continue
                if seconds < 0:
                    continue
                rendered["seconds"] = seconds
                if progress is not None:
                    progress(seconds)
            returncode = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            stop_event.set()
            reporter.join(timeout=1)
        if returncode != 0:
            stderr_file.seek(0)
            message = stderr_file.read().decode("utf-8", "replace").strip()
            raise RuntimeError(message[-4000:] or f"{description} failed")


def _encode_with_codec_fallback(build_command: Callable[[str], list[str]], **run_kwargs) -> str:
    """Encode with the configured codec and retry once with libx264, like the MoviePy path."""
    effective_codec = video._get_effective_video_codec()
    try:
        _run_ffmpeg(build_command(effective_codec), **run_kwargs)
        return effective_codec
    except Exception as exc:
        if effective_codec == video._DEFAULT_VIDEO_CODEC:
            raise
        _run_ffmpeg(build_command(video._DEFAULT_VIDEO_CODEC), **run_kwargs)
        video._disable_runtime_video_codec(effective_codec, str(exc))
        return video._DEFAULT_VIDEO_CODEC


def _staged_output_path(output_file: str) -> str:
    descriptor, staged = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_file)}.",
        suffix=os.path.splitext(output_file)[1] or ".mp4",
        dir=os.path.dirname(os.path.abspath(output_file)),
    )
    os.close(descriptor)
    return staged


def _publish(staged: str, output_file: str) -> None:
    if os.path.getsize(staged) == 0:
        raise RuntimeError("ffmpeg produced no output")
    os.replace(staged, output_file)


# ---------------------------------------------------------------------------
# combine


def _fit_filter(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    fit_mode: VideoFitMode,
) -> str:
    """Return the scale/crop/pad chain that matches ``video._fit_clip_to_canvas``."""
    if (source_width, source_height) == (target_width, target_height):
        return ""
    if source_width * target_height == source_height * target_width:
        return f"scale={target_width}:{target_height}"
    width_scale = target_width / source_width
    height_scale = target_height / source_height
    if fit_mode == VideoFitMode.cover:
        scale_factor = max(width_scale, height_scale)
        resized_width = max(target_width, math.ceil(source_width * scale_factor))
        resized_height = max(target_height, math.ceil(source_height * scale_factor))
        crop_x = max(0, (resized_width - target_width) // 2)
        crop_y = max(0, (resized_height - target_height) // 2)
        return (
            f"scale={resized_width}:{resized_height},"
            f"crop={target_width}:{target_height}:{crop_x}:{crop_y}"
        )
    scale_factor = min(width_scale, height_scale)
    resized_width = max(1, min(target_width, int(source_width * scale_factor)))
    resized_height = max(1, min(target_height, int(source_height * scale_factor)))
    pad_x = (target_width - resized_width) // 2
    pad_y = (target_height - resized_height) // 2
    return (
        f"scale={resized_width}:{resized_height},"
        f"pad={target_width}:{target_height}:{pad_x}:{pad_y}:black"
    )


def _plan_clips(
    video_paths: List[str],
    *,
    audio_duration: float,
    video_concat_mode: VideoConcatMode,
    max_clip_duration: int,
    clip_speed: float,
    source_usage: dict[str, int] | None,
    source_groups: dict[str, str] | None,
):
    """Choose slices exactly as ``video.combine_videos`` does, without rendering them.

    Returns ``(slices, required_duration)`` where each slice is
    ``(SubClippedVideoClip, output_duration)`` in playback order.
    """
    required_duration = video._get_required_video_duration(audio_duration)
    source_clip_duration = max_clip_duration * clip_speed
    subclipped_items: list[video.SubClippedVideoClip] = []
    later_items: dict[str, list[video.SubClippedVideoClip]] = {}
    for video_path in video_paths:
        try:
            width, height, clip_duration = _video_geometry(_probe(video_path))
            if not math.isfinite(clip_duration) or clip_duration <= 0 or width <= 0 or height <= 0:
                raise ValueError("invalid video duration or dimensions")
        except Exception as exc:
            logger.warning(f"skipping unreadable video source: path={video_path}, error={exc}")
            continue
        start_time = 0.0
        segments = []
        while start_time < clip_duration:
            end_time = min(start_time + source_clip_duration, clip_duration)
            if end_time > start_time:
                segments.append(
                    video.SubClippedVideoClip(
                        file_path=video_path,
                        start_time=start_time,
                        end_time=end_time,
                        width=width,
                        height=height,
                        source_file_path=video_path,
                    )
                )
            start_time = end_time
        if video_concat_mode.value == VideoConcatMode.sequential.value:
            subclipped_items.append(segments[0])
            later_items[video_path] = segments[1:]
        else:
            subclipped_items.extend(segments)

    subclipped_items = video._prioritize_unique_source_clips(
        subclipped_items=subclipped_items,
        concat_mode=video_concat_mode,
        **({"source_usage": source_usage, "source_groups": source_groups}
           if source_usage is not None else {}),
    )
    first_item_count = len(subclipped_items)
    if later_items:
        sources = dict.fromkeys(item.source_file_path for item in subclipped_items)
        rounds = itertools.zip_longest(*(later_items[source] for source in sources))
        subclipped_items += [item for row in rounds for item in row if item is not None]

    slices = []
    covered = 0.0
    for index, item in enumerate(subclipped_items):
        if covered >= required_duration:
            break
        if index >= first_item_count and covered >= audio_duration:
            break
        output_duration = min(max_clip_duration, (item.end_time - item.start_time) / clip_speed)
        slices.append((item, output_duration))
        covered += output_duration

    if slices and covered < required_duration:
        logger.warning(
            f"video duration ({covered:.2f}s) is shorter than required duration "
            f"({required_duration:.2f}s), looping clips to match audio length."
        )
        base = list(slices)
        for entry in itertools.cycle(base):
            if covered >= required_duration:
                break
            slices.append(entry)
            covered += entry[1]
    return slices, required_duration


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    clip_speed: float = 1.0,
    video_fit_mode: VideoFitMode = VideoFitMode.cover,
    source_usage: dict[str, int] | None = None,
    source_groups: dict[str, str] | None = None,
    used_video_paths: List[str] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    transition_value = getattr(video_transition_mode, "value", video_transition_mode)
    if transition_value not in _SUPPORTED_TRANSITIONS:
        raise UnsupportedRenderOption(f"transition {transition_value!r}")

    video_concat_mode = VideoConcatMode(video_concat_mode or VideoConcatMode.random)
    audio_duration = _media_duration(_probe(audio_file), "audio")
    logger.info(f"audio duration: {audio_duration} seconds")
    speed = utils.normalize_clip_speed(clip_speed)
    fit_mode = VideoFitMode(video_fit_mode)
    video_width, video_height = VideoAspect(video_aspect).to_resolution()

    slices, required_duration = _plan_clips(
        video_paths,
        audio_duration=audio_duration,
        video_concat_mode=video_concat_mode,
        max_clip_duration=max_clip_duration,
        clip_speed=speed,
        source_usage=source_usage,
        source_groups=source_groups,
    )
    if not slices:
        if video_paths:
            raise RuntimeError("no readable video clips available for merging")
        logger.warning("no clips available for merging")
        return combined_video_path
    logger.info(
        f"rendering {len(slices)} clips in one ffmpeg pass, "
        f"required duration: {required_duration:.2f}s"
    )

    def slice_filter(input_index: int, item, output_duration: float) -> str:
        played_duration = (item.end_time - item.start_time) / speed
        chain = [f"setpts=(PTS-STARTPTS)/{speed:g}", f"fps={video.fps}"]
        fit = _fit_filter(item.width, item.height, video_width, video_height, fit_mode)
        if fit:
            chain.append(fit)
        chain.append("setsar=1")
        fade_duration = min(1.0, played_duration)
        if transition_value == VideoTransitionMode.fade_in.value:
            chain.append(f"fade=t=in:st=0:d={fade_duration:.3f}")
        elif transition_value == VideoTransitionMode.fade_out.value:
            chain.append(
                f"fade=t=out:st={max(0.0, played_duration - fade_duration):.3f}:d={fade_duration:.3f}"
            )
        chain.append(f"trim=duration={output_duration:.3f},setpts=PTS-STARTPTS")
        return f"[{input_index}:v:0]{','.join(chain)}[v{input_index}]"

    # Slices that start after the narration ends are cut by the final -t anyway.
    timeline = []
    cursor = 0.0
    for item, output_duration in slices:
        if cursor >= audio_duration:
            break
        timeline.append((item, output_duration, cursor))
        cursor += output_duration

    # FFmpeg decodes the first frame of every input before it starts, so one
    # graph with every slice of a 15-minute video holds hundreds of 4K decoders
    # in memory. Chunks of a few slices keep memory flat. Every chunk is
    # encoded once with identical settings and joined without re-encoding.
    chunk_size = _get_combine_chunk_size()
    chunks = [timeline[i:i + chunk_size] for i in range(0, len(timeline), chunk_size)]
    output_dir = os.path.dirname(os.path.abspath(combined_video_path))
    work_dir = tempfile.mkdtemp(prefix=".ffmpeg-combine-", dir=output_dir)
    staged = _staged_output_path(combined_video_path)
    codec = video._get_effective_video_codec()
    try:
        chunk_files = []
        for chunk_index, chunk in enumerate(chunks):
            chunk_start = chunk[0][2]
            chunk_length = min(
                sum(entry[1] for entry in chunk), audio_duration - chunk_start
            )
            inputs: list[str] = []
            graph: list[str] = []
            for input_index, (item, output_duration, _start) in enumerate(chunk):
                inputs += [
                    "-ss", f"{item.start_time:.3f}",
                    "-t", f"{item.end_time - item.start_time:.3f}",
                    "-i", item.file_path,
                ]
                graph.append(slice_filter(input_index, item, output_duration))
            labels = "".join(f"[v{index}]" for index in range(len(chunk)))
            graph.append(
                f"{labels}concat=n={len(chunk)}:v=1:a=0,{video._BT709_VIDEO_FILTER}[vout]"
            )
            graph_file = os.path.join(work_dir, f"graph-{chunk_index:04d}.txt")
            with open(graph_file, "w", encoding="utf-8") as fp:
                fp.write(";\n".join(graph))
            chunk_file = os.path.join(work_dir, f"chunk-{chunk_index:04d}.mp4")

            def build_command(chunk_codec: str) -> list[str]:
                return [
                    utils.get_ffmpeg_binary(), "-nostdin", "-y", "-hide_banner",
                    "-loglevel", "error", "-progress", "pipe:1", "-nostats",
                    *inputs,
                    "-filter_complex_script", graph_file,
                    "-map", "[vout]",
                    "-c:v", chunk_codec, "-threads", str(threads or 2),
                    "-pix_fmt", "yuv420p", "-r", str(video.fps),
                    "-t", f"{chunk_length:.3f}", "-an",
                    chunk_file,
                ]

            def report(seconds: float, offset: float = chunk_start) -> None:
                video._report_clip_progress(
                    progress_callback, offset + seconds, required_duration
                )

            run_kwargs = dict(
                description=f"ffmpeg combine chunk {chunk_index + 1}/{len(chunks)}",
                total_seconds=chunk_length,
                progress=report,
            )
            try:
                _run_ffmpeg(build_command(codec), **run_kwargs)
            except Exception as exc:
                # Same codec rule as MoviePy: a hardware encoder that fails is
                # replaced by libx264. Only the first chunk can switch, because
                # the chunks are joined without re-encoding.
                if codec == video._DEFAULT_VIDEO_CODEC or chunk_index > 0:
                    raise
                _run_ffmpeg(build_command(video._DEFAULT_VIDEO_CODEC), **run_kwargs)
                video._disable_runtime_video_codec(codec, str(exc))
                codec = video._DEFAULT_VIDEO_CODEC
            chunk_files.append(chunk_file)
            logger.info(
                f"rendered chunk {chunk_index + 1}/{len(chunks)}: "
                f"{chunk_start + chunk_length:.1f} of {audio_duration:.1f}s covered"
            )

        list_file = os.path.join(work_dir, "chunks.txt")
        with open(list_file, "w", encoding="utf-8") as fp:
            for chunk_file in chunk_files:
                fp.write(f"file '{video._format_ffmpeg_concat_path(chunk_file)}'\n")
        _run_ffmpeg(
            [
                utils.get_ffmpeg_binary(), "-nostdin", "-y", "-hide_banner",
                "-loglevel", "error", "-progress", "pipe:1", "-nostats",
                "-f", "concat", "-safe", "0", "-i", list_file,
                "-map", "0:v:0", "-c", "copy", "-movflags", "+faststart",
                staged,
            ],
            description="ffmpeg join chunks",
        )
        _publish(staged, combined_video_path)
    finally:
        video.delete_files(staged)
        shutil.rmtree(work_dir, ignore_errors=True)

    video._report_clip_progress(progress_callback, required_duration, required_duration)
    if used_video_paths is not None:
        elapsed = 0.0
        for item, output_duration in slices:
            if elapsed >= audio_duration:
                break
            used_video_paths.append(item.source_file_path)
            elapsed += output_duration
    logger.info("video combining completed")
    return combined_video_path


# ---------------------------------------------------------------------------
# generate


def _subtitle_frame(clip, canvas_width: int, canvas_height: int) -> Image.Image:
    """Draw one positioned MoviePy subtitle clip onto a transparent full canvas."""
    rgb = np.asarray(clip.get_frame(0), dtype=np.uint8)
    if clip.mask is not None:
        alpha = np.clip(np.asarray(clip.mask.get_frame(0)) * 255, 0, 255).astype(np.uint8)
    else:
        alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    sprite = Image.fromarray(np.dstack([rgb[..., :3], alpha]), "RGBA")
    x, y = clip.pos(0)
    if x == "center":
        x = (canvas_width - sprite.width) / 2
    if y == "center":
        y = (canvas_height - sprite.height) / 2
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    # The canvas is empty, so a plain paste copies the sprite's alpha as is.
    canvas.paste(sprite, (int(x), int(y)))
    return canvas


def _write_subtitle_track(
    subtitle_path: str,
    params: VideoParams,
    font_path: str,
    video_width: int,
    video_height: int,
    work_dir: str,
) -> str | None:
    """Render cues to PNGs and return a concat-demuxer list that shows them in time."""
    subtitles = moviepy_subtitles(subtitle_path)
    if not subtitles:
        return None
    shaped_font = subtitle_shaping.video_font(font_path, (text for _, text in subtitles))
    blank = os.path.join(work_dir, "blank.png")
    Image.new("RGBA", (video_width, video_height), (0, 0, 0, 0)).save(blank)

    entries: list[tuple[str, float]] = []
    cursor = 0.0
    for index, item in enumerate(subtitles):
        (start, end), _text = item
        clip = video._create_subtitle_clip(
            item, params,
            font_path=font_path, shaped_font=shaped_font,
            video_width=video_width, video_height=video_height,
        )
        try:
            image = _subtitle_frame(clip, video_width, video_height)
        finally:
            video.close_clip(clip)
        frame_path = os.path.join(work_dir, f"cue-{index:05d}.png")
        image.save(frame_path, compress_level=1)
        if start > cursor:
            entries.append((blank, start - cursor))
        elif start < cursor:
            # Overlapping cues: MoviePy draws both; show the later one from here.
            start = cursor
        if end > start:
            entries.append((frame_path, end - start))
            cursor = end
    entries.append((blank, 1.0))

    list_path = os.path.join(work_dir, "subtitles.txt")
    with open(list_path, "w", encoding="utf-8") as fp:
        fp.write("ffconcat version 1.0\n")
        for path, duration in entries:
            fp.write(f"file '{video._format_ffmpeg_concat_path(path)}'\n")
            fp.write(f"duration {duration:.3f}\n")
        # The concat demuxer ignores the last duration unless the file repeats.
        fp.write(f"file '{video._format_ffmpeg_concat_path(blank)}'\n")
    return list_path


def _resolve_font_path(params: VideoParams) -> str:
    if not params.font_name:
        params.font_name = "STHeitiMedium.ttc"
    font_path = file_security.resolve_path_within_directory(utils.font_dir(), params.font_name)
    if os.name == "nt":
        font_path = font_path.replace("\\", "/")
    return font_path


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
    bgm_file_override: str | None = None,
) -> bool:
    if getattr(params, "subtitle_animation", "none") in _SPRING_ANIMATIONS and params.subtitle_enabled:
        raise UnsupportedRenderOption(f"subtitle animation {params.subtitle_animation!r}")

    video_width, video_height = VideoAspect(params.video_aspect).to_resolution()
    logger.info(f"generating video with ffmpeg: {video_width} x {video_height}")
    video_duration = _media_duration(_probe(video_path), "video")
    voice_info = _probe(audio_path)
    sample_rate = _audio_sample_rate(voice_info) or 44100

    bgm_enabled = bgm_service.should_use_bgm(params.bgm_type, params.bgm_volume)
    bgm_file = ""
    if bgm_enabled:
        bgm_file = (
            bgm_file_override
            if bgm_file_override is not None
            else video.get_bgm_file(bgm_type=params.bgm_type, bgm_file=params.bgm_file)
        )
    bgm_mix_succeeded = True
    bgm_duration = 0.0
    if bgm_file:
        # Same contract as MoviePy: a BGM that cannot be read degrades to a
        # narration-only video and reports False. It is checked before encoding.
        try:
            bgm_info = _probe(bgm_file)
            bgm_duration = _media_duration(bgm_info, "audio")
            if bgm_duration <= 0:
                raise ValueError("background music has no audio")
            sample_rate = max(sample_rate, _audio_sample_rate(bgm_info))
        except Exception:
            bgm_mix_succeeded = False
            logger.exception(
                f"failed to mix background music: type={params.bgm_type}, file={bgm_file}"
            )
            bgm_file = ""

    output_dir = os.path.dirname(os.path.abspath(output_file))
    work_dir = tempfile.mkdtemp(prefix=".ffmpeg-subtitles-", dir=output_dir)
    staged = _staged_output_path(output_file)
    try:
        subtitle_list = None
        if params.subtitle_enabled and subtitle_path and os.path.exists(subtitle_path):
            font_path = _resolve_font_path(params)
            subtitle_list = _write_subtitle_track(
                subtitle_path, params, font_path, video_width, video_height, work_dir
            )

        inputs = ["-i", video_path, "-i", audio_path]
        graph = []
        video_label = "0:v:0"
        if subtitle_list:
            inputs += ["-f", "concat", "-safe", "0", "-i", subtitle_list]
            graph.append(
                "[2:v]format=rgba[subs];"
                f"[0:v:0][subs]overlay=0:0:format=auto:eof_action=repeat,"
                f"{video._BT709_VIDEO_FILTER}[vout]"
            )
            video_label = "[vout]"
        voice_volume = 1.0 if params.voice_volume is None else params.voice_volume
        graph.append(f"[1:a]volume={voice_volume:g}[voice]")
        audio_label = "[voice]"
        if bgm_file:
            bgm_index = 3 if subtitle_list else 2
            if bgm_file_override is None:
                inputs += ["-stream_loop", "-1", "-i", bgm_file]
                bgm_length = video_duration
            else:
                inputs += ["-i", bgm_file]
                bgm_length = bgm_duration
            fade_start = max(0.0, bgm_length - _BGM_FADE_OUT_SECONDS)
            graph.append(
                f"[{bgm_index}:a]atrim=0:{bgm_length:.3f},volume={params.bgm_volume:g},"
                f"afade=t=out:st={fade_start:.3f}:d={_BGM_FADE_OUT_SECONDS:g}[bgm];"
                "[voice][bgm]amix=inputs=2:duration=longest:normalize=0[aout]"
            )
            audio_label = "[aout]"

        def build_command(codec: str) -> list[str]:
            video_args = ["-map", video_label]
            if video_label == "0:v:0":
                video_args += ["-vf", video._BT709_VIDEO_FILTER]
            return [
                utils.get_ffmpeg_binary(), "-nostdin", "-y", "-hide_banner",
                "-loglevel", "error", "-progress", "pipe:1", "-nostats",
                *inputs,
                "-filter_complex", ";".join(graph),
                *video_args,
                "-map", audio_label,
                "-c:v", codec, "-threads", str(params.n_threads or 2),
                "-pix_fmt", "yuv420p", "-r", str(video.fps),
                "-c:a", video.audio_codec, "-b:a", video.audio_bitrate,
                "-ar", str(sample_rate),
                "-t", f"{video_duration:.3f}",
                "-movflags", "+faststart",
                staged,
            ]

        _encode_with_codec_fallback(
            build_command,
            description="ffmpeg final video render",
            total_seconds=video_duration,
        )
        _publish(staged, output_file)
    finally:
        video.delete_files(staged)
        shutil.rmtree(work_dir, ignore_errors=True)
    return bgm_mix_succeeded
