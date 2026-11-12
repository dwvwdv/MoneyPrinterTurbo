"""Time one render backend on the sample inputs from ``make_assets.py``.

Runs the same two steps as ``task.generate_final_videos()`` (combine, then
generate) through the chosen backend, with a fixed random seed so every backend
picks the same material slices. Writes a JSON report with:

- ``clip_processing_s``: MoviePy only; cutting, fitting and encoding each slice.
- ``concat_s``: MoviePy only; the FFmpeg concat re-encode.
- ``combine_s``: the whole combine step (for FFmpeg it is one pass).
- ``final_encode_s``: the generate step (subtitles, narration, BGM, final encode).
- ``total_s``, peak memory, output durations/sizes and the contract values
  (progress samples, used_video_paths count, BGM bool).

Usage::

    python scripts/render_benchmark/run_benchmark.py --assets bench-assets \\
        --minutes 14 --backend ffmpeg --out bench-out
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.models.schema import VideoAspect, VideoParams  # noqa: E402
from app.services import render_backend, video  # noqa: E402


def _probe(path: str) -> dict:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(output)
    streams = {s["codec_type"]: s for s in info["streams"]}
    return {
        "duration_s": round(float(info["format"]["duration"]), 3),
        "size_mb": round(int(info["format"]["size"]) / 1024 / 1024, 2),
        "video": f"{streams['video']['codec_name']} {streams['video']['width']}x{streams['video']['height']}"
        if "video" in streams else None,
        "audio": f"{streams['audio']['codec_name']} {streams['audio'].get('sample_rate')}Hz"
        if "audio" in streams else None,
    }


def _peak_rss_mb() -> dict:
    scale = 1024 if platform.system() != "Darwin" else 1024 * 1024
    return {
        "python_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale, 1),
        "ffmpeg_peak_rss_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / scale, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets", required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--backend", choices=["moviepy", "ffmpeg"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=2, help="VideoParams.n_threads (default 2)")
    parser.add_argument("--aspect", default="9:16", choices=["9:16", "16:9", "1:1"])
    parser.add_argument("--transition", default=None)
    parser.add_argument("--no-subtitles", action="store_true")
    parser.add_argument("--no-bgm", action="store_true")
    parser.add_argument("--seed", type=int, default=1792)
    parser.add_argument(
        "--step", choices=["all", "combine", "generate"], default="all",
        help="run one step only (long MoviePy runs can be split); results merge into report.json",
    )
    args = parser.parse_args(argv)

    tag = f"{args.minutes:g}min"
    narration = os.path.join(args.assets, f"narration-{tag}.mp3")
    subtitles = os.path.join(args.assets, f"subtitles-{tag}.srt")
    materials = sorted(glob.glob(os.path.join(args.assets, "materials", "*.mp4")))
    run_dir = os.path.join(args.out, f"{args.backend}-{tag}")
    os.makedirs(run_dir, exist_ok=True)
    combined = os.path.join(run_dir, "combined-1.mp4")
    final = os.path.join(run_dir, "final-1.mp4")

    params = VideoParams(
        video_subject="render benchmark",
        video_aspect=VideoAspect(args.aspect),
        video_transition_mode=args.transition,
        subtitle_enabled=not args.no_subtitles,
        bgm_type="" if args.no_bgm else "random",
        n_threads=args.threads,
    )
    backend = render_backend.get_render_backend(args.backend)

    timings: dict[str, float] = {}
    progress_samples: list[tuple[float, float]] = []
    started = time.perf_counter()

    def on_progress(fraction: float) -> None:
        progress_samples.append((round(time.perf_counter() - started, 2), round(fraction, 4)))

    # Split MoviePy's combine into clip processing and concat by timing the concat call.
    original_concat = video.concat_video_clips_with_ffmpeg

    def timed_concat(*a, **kw):
        concat_started = time.perf_counter()
        timings["clip_processing_s"] = concat_started - started
        try:
            return original_concat(*a, **kw)
        finally:
            timings["concat_s"] = time.perf_counter() - concat_started

    video.concat_video_clips_with_ffmpeg = timed_concat

    report_path = os.path.join(run_dir, "report.json")
    previous = {}
    if args.step != "all" and os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as fp:
            previous = json.load(fp)
    contracts = dict(previous.get("contracts", {}))

    if args.step in ("all", "combine"):
        random.seed(args.seed)
        used_video_paths: list[str] = []
        backend.combine_videos(
            combined_video_path=combined,
            video_paths=materials,
            audio_file=narration,
            video_aspect=params.video_aspect,
            video_concat_mode=params.video_concat_mode,
            video_transition_mode=params.video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            clip_speed=params.video_clip_speed,
            video_fit_mode=params.video_fit_mode,
            source_usage={},
            used_video_paths=used_video_paths,
            progress_callback=on_progress,
        )
        timings["combine_s"] = time.perf_counter() - started
        contracts.update({
            "progress_calls": len(progress_samples),
            "progress_first": progress_samples[:1],
            "progress_last": progress_samples[-1:],
            "used_video_paths": len(used_video_paths),
        })

    if args.step in ("all", "generate"):
        generate_started = time.perf_counter()
        contracts["bgm_mix_succeeded"] = backend.generate_video(
            video_path=combined,
            audio_path=narration,
            subtitle_path=subtitles,
            output_file=final,
            params=params,
        )
        timings["final_encode_s"] = time.perf_counter() - generate_started

    merged_timings = {**previous.get("timings", {}), **{k: round(v, 1) for k, v in timings.items()}}
    merged_timings["total_s"] = round(
        merged_timings.get("combine_s", 0) + merged_timings.get("final_encode_s", 0), 1
    )
    memory = _peak_rss_mb()
    for key, value in previous.get("memory", {}).items():
        memory[key] = max(memory.get(key, 0), value)

    report = {
        "backend": args.backend,
        "minutes": args.minutes,
        "materials": len(materials),
        "params": {
            "aspect": args.aspect, "n_threads": args.threads, "transition": args.transition,
            "subtitles": not args.no_subtitles, "bgm": not args.no_bgm,
            "video_clip_concurrency": video._get_clip_processing_concurrency(),
        },
        "machine": {
            "cpus": os.cpu_count(), "platform": platform.platform(),
            "python": platform.python_version(),
            "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
        },
        "timings": merged_timings,
        "memory": memory,
        "contracts": contracts,
        "combined": _probe(combined) if os.path.exists(combined) else None,
        "final": _probe(final) if os.path.exists(final) else None,
    }
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
