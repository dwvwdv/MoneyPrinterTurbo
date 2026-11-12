"""The render backend seam must keep the MoviePy rendering contract unchanged."""

import os
import tempfile
import unittest
from unittest.mock import patch

from app.models.schema import VideoParams
from app.services import render_backend
from app.services import task as tm


class TestMoviePyBackend(unittest.TestCase):
    def test_default_backend_is_moviepy(self):
        backend = render_backend.get_render_backend()

        self.assertIsInstance(backend, render_backend.MoviePyBackend)
        self.assertIsInstance(backend, render_backend.BaseRenderBackend)
        self.assertEqual(backend.name, "moviepy")

    def test_backend_missing_a_step_cannot_be_instantiated(self):
        class CombineOnly(render_backend.BaseRenderBackend):
            name = "combine-only"

            def combine_videos(self, *args, **kwargs):
                return ""

        with self.assertRaises(TypeError):
            CombineOnly()

    def test_combine_videos_forwards_every_argument_and_the_result(self):
        backend = render_backend.MoviePyBackend()
        used_video_paths = []
        source_usage = {"a.mp4": 1}

        def progress_callback(fraction):
            return None

        with patch.object(
            render_backend.video, "combine_videos", return_value="combined-1.mp4"
        ) as combine:
            result = backend.combine_videos(
                combined_video_path="combined-1.mp4",
                video_paths=["a.mp4", "b.mp4"],
                audio_file="audio.mp3",
                clip_speed=1.5,
                source_usage=source_usage,
                used_video_paths=used_video_paths,
                progress_callback=progress_callback,
            )

        self.assertEqual(result, "combined-1.mp4")
        kwargs = combine.call_args.kwargs
        # Batch allocation reads back the caller's own list and dict, and task
        # progress depends on the caller's callback, so identity must survive.
        self.assertIs(kwargs["used_video_paths"], used_video_paths)
        self.assertIs(kwargs["source_usage"], source_usage)
        self.assertIs(kwargs["progress_callback"], progress_callback)
        self.assertEqual(kwargs["clip_speed"], 1.5)

    def test_generate_video_returns_the_bgm_result_unchanged(self):
        backend = render_backend.MoviePyBackend()
        params = VideoParams(video_subject="test")

        for bgm_result in (True, False):
            with self.subTest(bgm_result=bgm_result), patch.object(
                render_backend.video, "generate_video", return_value=bgm_result
            ) as generate:
                result = backend.generate_video(
                    video_path="combined-1.mp4",
                    audio_path="audio.mp3",
                    subtitle_path="",
                    output_file="final-1.mp4",
                    params=params,
                    bgm_file_override="bgm.mp3",
                )

            self.assertIs(result, bgm_result)
            self.assertEqual(generate.call_args.kwargs["bgm_file_override"], "bgm.mp3")


class _RecordingBackend(render_backend.BaseRenderBackend):
    name = "recording"

    def __init__(self):
        self.calls = []

    def combine_videos(self, **kwargs):
        self.calls.append(("combine", kwargs))
        kwargs["used_video_paths"].append(kwargs["video_paths"][0])
        kwargs["progress_callback"](0.5)
        return kwargs["combined_video_path"]

    def generate_video(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return True


class TestGenerateFinalVideosUsesBackend(unittest.TestCase):
    def test_task_renders_through_the_selected_backend(self):
        backend = _RecordingBackend()
        params = VideoParams(
            video_subject="test", video_count=2, video_source="local",
            bgm_type="", match_materials_to_script=False,
        )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(tm.render_backend, "get_render_backend", return_value=backend),
            patch.object(tm.video, "combine_videos") as moviepy_combine,
            patch.object(tm.video, "generate_video") as moviepy_generate,
            patch.object(tm.utils, "task_dir", return_value=directory),
            patch.object(tm.task_artifacts, "patch_script_data") as patch_script_data,
            patch.object(tm.sm.state, "update_task") as update_task,
        ):
            final_paths, combined_paths, warnings = tm.generate_final_videos(
                task_id="backend-task",
                params=params,
                downloaded_videos=["material.mp4"],
                audio_file="audio.mp3",
                subtitle_path="",
                audio_duration=5,
            )

        moviepy_combine.assert_not_called()
        moviepy_generate.assert_not_called()
        self.assertEqual(
            [stage for stage, _ in backend.calls],
            ["combine", "generate", "combine", "generate"],
        )
        self.assertEqual(final_paths, [
            os.path.join(directory, "final-1.mp4"),
            os.path.join(directory, "final-2.mp4"),
        ])
        self.assertEqual(combined_paths, [
            os.path.join(directory, "combined-1.mp4"),
            os.path.join(directory, "combined-2.mp4"),
        ])
        # The second video reuses the only material, which batch tracking
        # must still notice through the backend.
        self.assertEqual(warnings, [
            {"code": "batch_materials_reused", "video_index": 2, "count": 1},
        ])
        self.assertEqual(
            patch_script_data.call_args.kwargs["material_selections"][1]["reused_files"],
            ["material.mp4"],
        )
        # Halfway through the first combine maps to 50 + 12.5 / 2.
        progress_values = [
            call.kwargs["progress"] for call in update_task.call_args_list
            if "progress" in call.kwargs
        ]
        self.assertIn(56.25, progress_values)


if __name__ == "__main__":
    unittest.main()
