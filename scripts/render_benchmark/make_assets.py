"""Generate deterministic sample inputs for the render backend benchmark.

Everything is synthesized with FFmpeg, so anyone can rebuild the exact same
inputs without downloading stock footage:

- ``materials/``: source clips in the mix that stock sites return (4K and
  1080p landscape, 1080p portrait), each with moving content and film grain
  so the encoder does real work.
- ``narration.mp3``: a narration stand-in of the requested length.
- ``subtitles.srt``: one cue every few seconds, mixed Chinese/English text.

Usage::

    python scripts/render_benchmark/make_assets.py --out bench-assets --minutes 14
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# (width, height) cycle: 4K landscape, 1080p landscape, 1080p portrait.
_SOURCE_SIZES = [(3840, 2160), (1920, 1080), (1080, 1920)]
_SOURCE_PATTERNS = ["testsrc2", "smptehdbars", "mandelbrot", "life"]
_SUBTITLE_LINES = [
    "这是一段用来测试渲染速度的字幕",
    "Long videos expose the cost of every extra encode",
    "每一段素材都会被裁剪、缩放，再拼接成完整的画面",
    "The narration keeps going for the whole video",
    "字幕需要和旁白保持同步，并且不能被裁切",
    "Background music is mixed under the voice track",
]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _source_filter(pattern: str, width: int, height: int, duration: float) -> str:
    if pattern == "life":
        base = f"life=s={width // 4}x{height // 4}:r=30:mold=10:ratio=0.3,scale={width}:{height}:flags=neighbor"
    elif pattern == "mandelbrot":
        # mandelbrot is slow at 4K; render small and upscale.
        base = f"mandelbrot=s={width // 4}x{height // 4}:r=30,scale={width}:{height}"
    else:
        base = f"{pattern}=s={width}x{height}:r=30"
    # Grain and a slow pan keep every frame different, like real footage.
    return (
        f"{base},trim=duration={duration},"
        "noise=alls=4:allf=t,"
        "format=yuv420p"
    )


def make_materials(out_dir: str, count: int, duration: float) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for index in range(count):
        width, height = _SOURCE_SIZES[index % len(_SOURCE_SIZES)]
        pattern = _SOURCE_PATTERNS[index % len(_SOURCE_PATTERNS)]
        path = os.path.join(out_dir, f"material-{index:03d}-{width}x{height}.mp4")
        paths.append(path)
        if os.path.exists(path):
            continue
        print(f"material {index + 1}/{count}: {os.path.basename(path)}", flush=True)
        _run([
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", _source_filter(pattern, width, height, duration),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-g", "60", "-an", path,
        ])
    return paths


def make_narration(path: str, seconds: float) -> None:
    if os.path.exists(path):
        return
    # A tone that wanders in pitch plus soft noise, so AAC/MP3 encode real content.
    _run([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=24000:duration={seconds}",
        "-f", "lavfi", "-i", f"anoisesrc=color=pink:amplitude=0.05:sample_rate=24000:duration={seconds}",
        "-filter_complex", "[0][1]amix=inputs=2:normalize=0,vibrato=f=0.5:d=0.6",
        "-ac", "1", "-c:a", "libmp3lame", "-b:a", "48k", path,
    ])


def _srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def make_subtitles(path: str, seconds: float, cue_seconds: float = 3.2) -> int:
    cues = []
    start = 0.0
    index = 0
    while start < seconds - 0.5:
        end = min(seconds, start + cue_seconds - 0.2)
        text = _SUBTITLE_LINES[index % len(_SUBTITLE_LINES)]
        cues.append(f"{index + 1}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
        start += cue_seconds
        index += 1
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(cues))
    return len(cues)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--minutes", type=float, default=14.0)
    parser.add_argument("--materials", type=int, default=48)
    parser.add_argument("--material-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)

    seconds = args.minutes * 60
    os.makedirs(args.out, exist_ok=True)
    make_materials(os.path.join(args.out, "materials"), args.materials, args.material_seconds)
    tag = f"{args.minutes:g}min"
    make_narration(os.path.join(args.out, f"narration-{tag}.mp3"), seconds)
    cues = make_subtitles(os.path.join(args.out, f"subtitles-{tag}.srt"), seconds)
    print(f"assets ready in {args.out}: {args.materials} materials, {cues} subtitle cues")
    return 0


if __name__ == "__main__":
    sys.exit(main())
