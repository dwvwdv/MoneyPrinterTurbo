"""The FFmpeg render backend must keep the contracts of the MoviePy backend."""

import os
import random
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from app.models.schema import VideoFitMode, VideoParams, VideoTransitionMode
from app.services import render_backend, render_ffmpeg

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _make_video(path, size, seconds):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc2=s={size}:r=30:d={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path],
        check=True,
    )


def _make_audio(path, seconds):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"sine=frequency=330:sample_rate=24000:duration={seconds}", path],
        check=True,
    )


def _duration(path):
    return float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True,
    ).stdout)


class TestBackendSelection(unittest.TestCase):
    def test_moviepy_stays_the_default(self):
        with patch.dict(render_backend.config.app, {}, clear=False):
            render_backend.config.app.pop("render_backend", None)
            self.assertEqual(render_backend.get_render_backend().name, "moviepy")

    def test_config_selects_ffmpeg(self):
        with patch.dict(render_backend.config.app, {"render_backend": "ffmpeg"}):
            backend = render_backend.get_render_backend()
        self.assertIsInstance(backend, render_backend.FFmpegBackend)
        self.assertIsInstance(backend, render_backend.BaseRenderBackend)

    def test_unknown_name_uses_moviepy(self):
        self.assertEqual(render_backend.get_render_backend("gpu-magic").name, "moviepy")


class TestFallbackBeforeRendering(unittest.TestCase):
    def setUp(self):
        self.backend = render_backend.get_render_backend("ffmpeg")

    def test_unsupported_transition_renders_with_moviepy(self):
        with patch.object(render_backend.video, "combine_videos", return_value="c.mp4") as moviepy, \
                patch.object(render_ffmpeg, "_probe") as probe:
            result = self.backend.combine_videos(
                combined_video_path="c.mp4", video_paths=["a.mp4"], audio_file="a.mp3",
                video_transition_mode=VideoTransitionMode.slide_in,
            )
        self.assertEqual(result, "c.mp4")
        moviepy.assert_called_once()
        # Nothing was probed or rendered by FFmpeg before handing over.
        probe.assert_not_called()

    def test_spring_subtitles_render_with_moviepy(self):
        params = VideoParams(video_subject="x", subtitle_animation="pop_spring")
        with patch.object(render_backend.video, "generate_video", return_value=False) as moviepy:
            result = self.backend.generate_video("c.mp4", "a.mp3", "s.srt", "f.mp4", params)
        self.assertIs(result, False)
        moviepy.assert_called_once()

    def test_missing_ffmpeg_renders_with_moviepy(self):
        with patch.object(self.backend, "_ffmpeg_available", return_value=False), \
                patch.object(render_backend.video, "combine_videos", return_value="c.mp4") as moviepy:
            self.backend.combine_videos("c.mp4", ["a.mp4"], "a.mp3")
        moviepy.assert_called_once()

    def test_missing_ffprobe_renders_with_moviepy(self):
        # imageio-ffmpeg ships ffmpeg without ffprobe.
        with patch.object(render_ffmpeg, "ffprobe_binary", return_value=None), \
                patch.object(render_ffmpeg, "_probe") as probe, \
                patch.object(render_backend.video, "combine_videos", return_value="c.mp4") as moviepy:
            self.backend.combine_videos("c.mp4", ["a.mp4"], "a.mp3")
        moviepy.assert_called_once()
        probe.assert_not_called()

    def test_failure_after_start_is_not_rerendered(self):
        with patch.object(render_ffmpeg, "combine_videos", side_effect=RuntimeError("encoder died")), \
                patch.object(render_backend.video, "combine_videos") as moviepy:
            with self.assertRaises(RuntimeError):
                self.backend.combine_videos("c.mp4", ["a.mp4"], "a.mp3")
        moviepy.assert_not_called()


class TestFitFilter(unittest.TestCase):
    def test_same_size_needs_no_filter(self):
        self.assertEqual(render_ffmpeg._fit_filter(1080, 1920, 1080, 1920, VideoFitMode.cover), "")

    def test_cover_crops_the_centre(self):
        self.assertEqual(
            render_ffmpeg._fit_filter(1920, 1080, 1080, 1920, VideoFitMode.cover),
            "scale=3414:1920,crop=1080:1920:1167:0",
        )

    def test_contain_letterboxes(self):
        self.assertEqual(
            render_ffmpeg._fit_filter(1920, 1080, 1080, 1920, VideoFitMode.contain),
            "scale=1080:607,pad=1080:1920:0:656:black",
        )


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg is not installed")
class TestRenderWithFFmpeg(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.materials = []
        for index, size in enumerate(["640x360", "360x640", "480x480"]):
            path = os.path.join(self.tmp, f"m{index}.mp4")
            _make_video(path, size, 4)
            self.materials.append(path)
        self.audio = os.path.join(self.tmp, "voice.mp3")
        _make_audio(self.audio, 7)

    def test_combine_keeps_progress_and_material_tracking(self):
        combined = os.path.join(self.tmp, "combined-1.mp4")
        used, progress = [], []
        random.seed(1)
        with patch.dict(render_ffmpeg.config.app, {"ffmpeg_combine_chunk_size": 2}):
            render_ffmpeg.combine_videos(
                combined, self.materials, self.audio, max_clip_duration=2,
                used_video_paths=used, progress_callback=progress.append,
            )
        self.assertAlmostEqual(_duration(combined), _duration(self.audio), delta=0.1)
        self.assertTrue(progress)
        self.assertTrue(all(0 <= value <= 1 for value in progress))
        self.assertEqual(progress[-1], 1.0)
        self.assertEqual(len(used), 4)
        self.assertTrue(set(used) <= set(self.materials))
        # Temporary chunks, graphs and staged outputs are all removed.
        self.assertEqual(sorted(os.listdir(self.tmp)), sorted(
            [os.path.basename(p) for p in self.materials] + ["voice.mp3", "combined-1.mp4"]
        ))

    def test_unreadable_bgm_degrades_to_narration_only(self):
        combined = os.path.join(self.tmp, "combined-1.mp4")
        render_ffmpeg.combine_videos(combined, self.materials, self.audio, max_clip_duration=2)
        broken_bgm = os.path.join(self.tmp, "broken.mp3")
        with open(broken_bgm, "wb") as fp:
            fp.write(b"not audio")
        final = os.path.join(self.tmp, "final-1.mp4")
        params = VideoParams(video_subject="x", subtitle_enabled=False, bgm_type="random")
        ok = render_ffmpeg.generate_video(
            combined, self.audio, "", final, params, bgm_file_override=broken_bgm
        )
        self.assertIs(ok, False)
        self.assertAlmostEqual(_duration(final), _duration(combined), delta=0.1)

    def test_generate_draws_subtitles_and_mixes_bgm(self):
        combined = os.path.join(self.tmp, "combined-1.mp4")
        render_ffmpeg.combine_videos(combined, self.materials, self.audio, max_clip_duration=2)
        srt = os.path.join(self.tmp, "s.srt")
        with open(srt, "w", encoding="utf-8") as fp:
            fp.write("1\n00:00:00,500 --> 00:00:02,000\n你好 world\n\n"
                     "2\n00:00:03,000 --> 00:00:05,000\nsecond line\n")
        bgm = os.path.join(self.tmp, "bgm.mp3")
        _make_audio(bgm, 3)
        final = os.path.join(self.tmp, "final-1.mp4")
        params = VideoParams(video_subject="x", bgm_type="random", bgm_volume=0.2)
        ok = render_ffmpeg.generate_video(combined, self.audio, srt, final, params, bgm_file_override=bgm)
        self.assertIs(ok, True)
        self.assertAlmostEqual(_duration(final), _duration(combined), delta=0.1)
        # MoviePy reads audio at 44.1 kHz, so its final videos are 44.1 kHz.
        sample_rate = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate", "-of", "default=nw=1:nk=1", final],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        self.assertEqual(sample_rate, "44100")


if __name__ == "__main__":
    unittest.main()


class TestOverlappingSubtitles(unittest.TestCase):
    def test_cue_inside_another_cue_is_kept(self):
        # MoviePy draws both cues at their own times; the FFmpeg track must too.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        srt = os.path.join(tmp, "s.srt")
        with open(srt, "w", encoding="utf-8") as fp:
            fp.write("1\n00:00:00,000 --> 00:00:05,000\nlong cue\n\n"
                     "2\n00:00:01,000 --> 00:00:02,000\nshort cue\n")
        params = VideoParams(video_subject="x")
        font_path = render_ffmpeg._resolve_font_path(params)
        list_path = render_ffmpeg._write_subtitle_track(srt, params, font_path, 360, 640, tmp)
        with open(list_path, encoding="utf-8") as fp:
            lines = fp.read().splitlines()
        files = [os.path.basename(line.split("'")[1]) for line in lines if line.startswith("file ")]
        durations = [float(line.split()[1]) for line in lines if line.startswith("duration ")]
        self.assertEqual(files[:3], ["cue-00000.png", "overlap-00000.png", "cue-00000.png"])
        self.assertEqual(durations[:3], [1.0, 1.0, 3.0])
