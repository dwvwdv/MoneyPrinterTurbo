"""Render backend interface for the two video rendering entry points.

``task.generate_final_videos()`` renders each video in two steps:

1. ``combine_videos()`` cuts and joins the material clips into
   ``combined-N.mp4`` to match the narration length.
2. ``generate_video()`` adds narration, subtitles and background music to
   produce ``final-N.mp4``.

The steps stay separate on purpose: video-music providers such as Sonilo and
ElevenLabs read ``combined-N.mp4`` between them to compose the BGM.

A backend implements both steps with the same signatures (including
defaults) and contracts as the functions in ``app.services.video``; a test
keeps the three signatures in sync:

- ``combine_videos`` calls ``progress_callback``, when given, with a fraction
  in ``[0, 1]`` while it processes clips.
- ``source_usage``, ``source_groups`` and ``used_video_paths`` are passed only
  when the task allocates materials across a batch of videos, and are
  ``None`` otherwise. When ``used_video_paths`` is given, ``combine_videos``
  appends every material file it uses to that list (the caller's own list,
  mutated in place) so reuse can be tracked across videos; ``source_usage``
  and ``source_groups`` are read-only inputs for that allocation.
- ``generate_video`` returns whether background music was handled: ``True``
  when no BGM was requested or it mixed successfully, ``False`` when BGM was
  requested but failed and a narration-only video was written instead.

A backend is named after its rendering engine, not where it runs: an FFmpeg
backend would also render on this machine, and a remote worker could run
either engine. Only the MoviePy backend exists today, and it is always used,
so behavior is unchanged. Discovering and selecting other backends is left to
a later change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List

from app.models.schema import (
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services import video

DEFAULT_RENDER_BACKEND = "moviepy"


class BaseRenderBackend(ABC):
    """Renders the combined and final videos for one task.

    Subclasses set ``name`` to their rendering engine and implement both steps.
    """

    name: str

    @abstractmethod
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
    ) -> str:
        """Join the material clips into ``combined_video_path`` and return it."""

    @abstractmethod
    def generate_video(
        self,
        video_path: str,
        audio_path: str,
        subtitle_path: str,
        output_file: str,
        params: VideoParams,
        bgm_file_override: str | None = None,
    ) -> bool:
        """Write ``output_file`` and return whether BGM was handled."""


class MoviePyBackend(BaseRenderBackend):
    """The existing in-process MoviePy renderer."""

    name = DEFAULT_RENDER_BACKEND

    # Look the functions up on the module at call time, so patches of
    # ``video.combine_videos`` / ``video.generate_video`` still apply.
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
    ) -> str:
        return video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=video_paths,
            audio_file=audio_file,
            video_aspect=video_aspect,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=max_clip_duration,
            threads=threads,
            clip_speed=clip_speed,
            video_fit_mode=video_fit_mode,
            source_usage=source_usage,
            source_groups=source_groups,
            used_video_paths=used_video_paths,
            progress_callback=progress_callback,
        )

    def generate_video(
        self,
        video_path: str,
        audio_path: str,
        subtitle_path: str,
        output_file: str,
        params: VideoParams,
        bgm_file_override: str | None = None,
    ) -> bool:
        return video.generate_video(
            video_path=video_path,
            audio_path=audio_path,
            subtitle_path=subtitle_path,
            output_file=output_file,
            params=params,
            bgm_file_override=bgm_file_override,
        )


_DEFAULT_BACKEND = MoviePyBackend()


def get_render_backend() -> BaseRenderBackend:
    """Return the backend that renders task videos (currently always MoviePy)."""
    return _DEFAULT_BACKEND
