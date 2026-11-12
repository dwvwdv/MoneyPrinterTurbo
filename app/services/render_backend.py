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

A backend is named after its rendering engine, not where it runs: the FFmpeg
backend also renders on this machine, and a remote worker could run either
engine. MoviePy stays the default; ``render_backend = "ffmpeg"`` under
``[app]`` opts in to the FFmpeg backend.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from typing import Callable, List

from loguru import logger

from app.models.schema import (
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoParams,
    VideoTransitionMode,
)
from app.config import config
from app.services import render_ffmpeg, video
from app.utils import utils

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
    def combine_videos(self, *args, **kwargs) -> str:
        return video.combine_videos(*args, **kwargs)

    def generate_video(self, *args, **kwargs) -> bool:
        return video.generate_video(*args, **kwargs)


class FFmpegBackend(BaseRenderBackend):
    """Renders each step as one FFmpeg filter graph and a single encode.

    Falls back to MoviePy only before a step starts rendering: when FFmpeg is
    missing or the step uses an option this backend cannot draw. A failure
    after FFmpeg has started is raised, so a step never renders twice.
    """

    name = "ffmpeg"

    def __init__(self, fallback: BaseRenderBackend):
        self._fallback = fallback

    def _ffmpeg_available(self) -> bool:
        # Both are needed: the bundled imageio-ffmpeg binary has no ffprobe.
        binary = utils.get_ffmpeg_binary()
        has_ffmpeg = bool(shutil.which(binary) or os.path.isfile(binary))
        return has_ffmpeg and render_ffmpeg.ffprobe_binary() is not None

    def _run(self, step: str, *args, **kwargs):
        if not self._ffmpeg_available():
            logger.warning(f"ffmpeg or ffprobe not found, rendering {step} with moviepy")
            return getattr(self._fallback, step)(*args, **kwargs)
        try:
            return getattr(render_ffmpeg, step)(*args, **kwargs)
        except render_ffmpeg.UnsupportedRenderOption as exc:
            # Raised before FFmpeg starts, so nothing has been rendered yet.
            logger.info(f"ffmpeg backend does not support {exc}, rendering {step} with moviepy")
            return getattr(self._fallback, step)(*args, **kwargs)

    def combine_videos(self, *args, **kwargs) -> str:
        return self._run("combine_videos", *args, **kwargs)

    def generate_video(self, *args, **kwargs) -> bool:
        return self._run("generate_video", *args, **kwargs)


_MOVIEPY_BACKEND = MoviePyBackend()
_BACKENDS: dict[str, BaseRenderBackend] = {
    MoviePyBackend.name: _MOVIEPY_BACKEND,
    FFmpegBackend.name: FFmpegBackend(fallback=_MOVIEPY_BACKEND),
}


def get_render_backend(name: str | None = None) -> BaseRenderBackend:
    """Return the backend named by ``name`` or ``[app] render_backend`` (MoviePy by default)."""
    if name is None:
        name = config.app.get("render_backend", DEFAULT_RENDER_BACKEND)
    key = str(name or DEFAULT_RENDER_BACKEND).strip().lower()
    backend = _BACKENDS.get(key)
    if backend is None:
        logger.warning(f"unknown render backend {name!r}, using {DEFAULT_RENDER_BACKEND}")
        return _MOVIEPY_BACKEND
    return backend
