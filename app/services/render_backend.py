"""Render backend interface for the two video rendering entry points.

``task.generate_final_videos()`` renders each video in two steps:

1. ``combine_videos()`` cuts and joins the material clips into
   ``combined-N.mp4`` to match the narration length.
2. ``generate_video()`` adds narration, subtitles and background music to
   produce ``final-N.mp4``.

The steps stay separate on purpose: video-music providers such as Sonilo and
ElevenLabs read ``combined-N.mp4`` between them to compose the BGM.

A backend implements both steps with the same signatures and contracts as the
functions in ``app.services.video``:

- ``combine_videos`` calls ``progress_callback`` with a fraction in ``[0, 1]``
  while it processes clips, and appends every material file it uses to
  ``used_video_paths`` (the caller's list, mutated in place) so batch material
  allocation can track reuse across videos. ``source_usage`` and
  ``source_groups`` are read-only inputs for that allocation.
- ``generate_video`` returns whether background music was handled: ``True``
  when no BGM was requested or it mixed successfully, ``False`` when BGM was
  requested but failed and a narration-only video was written instead.

Only the local MoviePy backend exists today, and it is always used, so
behavior is unchanged. Discovering and selecting other backends is left to a
later change.
"""

from __future__ import annotations

from typing import Callable, List, Protocol, runtime_checkable

from app.models.schema import (
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import video


@runtime_checkable
class RenderBackend(Protocol):
    """Renders the combined and final videos for one task."""

    name: str

    def combine_videos(
        self,
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
    ) -> str: ...

    def generate_video(
        self,
        video_path: str,
        audio_path: str,
        subtitle_path: str,
        output_file: str,
        params: VideoParams,
        bgm_file_override: str | None = None,
    ) -> bool: ...


class LocalMoviePyBackend:
    """The existing in-process MoviePy/FFmpeg renderer."""

    name = "local"

    # Look the functions up on the module at call time, so patches of
    # ``video.combine_videos`` / ``video.generate_video`` still apply.
    def combine_videos(self, *args, **kwargs) -> str:
        return video.combine_videos(*args, **kwargs)

    def generate_video(self, *args, **kwargs) -> bool:
        return video.generate_video(*args, **kwargs)


_LOCAL_BACKEND = LocalMoviePyBackend()


def get_render_backend() -> RenderBackend:
    """Return the backend that renders task videos (currently always local)."""
    return _LOCAL_BACKEND
