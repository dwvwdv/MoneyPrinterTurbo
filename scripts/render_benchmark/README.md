# Render backend benchmark

Reproducible before/after timings for the `moviepy` and `ffmpeg` render backends.

```bash
# 1. Synthesize inputs: 48 materials (4K/1080p landscape, 1080p portrait, 20 s each),
#    a 14-minute narration and 263 Chinese/English subtitle cues.
python scripts/render_benchmark/make_assets.py --out bench-assets --minutes 14

# 2. Time each backend. Same seed, so both pick the same material slices.
python scripts/render_benchmark/run_benchmark.py --assets bench-assets --minutes 14 --backend ffmpeg --out bench-out
python scripts/render_benchmark/run_benchmark.py --assets bench-assets --minutes 14 --backend moviepy --out bench-out
# A long MoviePy run can be split; the second call merges into the same report:
#   ... --backend moviepy --step combine   then   ... --backend moviepy --step generate
```

Settings: portrait 1080x1920, `n_threads=2`, `video_clip_concurrency=1` (defaults),
no transition, subtitles on, random BGM, libx264.

## Results: 14-minute video, 4 vCPU, 15 GB RAM, FFmpeg 6.1.1

| step | MoviePy | FFmpeg | speed-up |
|---|---:|---:|---:|
| clip processing | 4201.7 s | n/a (chunked filter graphs) | |
| concat | 943.4 s | n/a (joined with `-c copy`) | |
| **combine total** | **5145.5 s (85.8 min)** | **1163.7 s (19.4 min)** | **4.4×** |
| **final encode** | **1441.3 s (24.0 min)** | **1038.7 s (17.3 min)** | **1.4×** |
| **total** | **6586.8 s (109.8 min)** | **2202.4 s (36.7 min)** | **3.0×** |
| peak RSS: Python / FFmpeg | 689 / 995 MB | 95 / 1543 MB | |

Output check: both finals are 840.07 s at 1080x1920 H.264. SSIM between them is 0.983
(the inputs carry random grain, so re-encodes never match bit for bit) and loudness is
−22.1 vs −22.3 LUFS. Contracts: both report 169 `used_video_paths` and `bgm_mix_succeeded=True`,
and progress ends at 1.0 (MoviePy: 169 callbacks, one per clip; FFmpeg: 1985, from `-progress`).

Raw reports: [`results/14min-moviepy.json`](results/14min-moviepy.json),
[`results/14min-ffmpeg.json`](results/14min-ffmpeg.json). The MoviePy run was split into
`--step combine` and `--step generate`, so its peak memory is the larger of the two steps.
The FFmpeg final report shows 48 kHz audio; that build was fixed afterwards to write 44.1 kHz like MoviePy.
