"""Utilities for managing Coqui TTS models."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

from TTS.api import TTS


class TTSModelError(RuntimeError):
    """Raised when a model or synthesis operation fails."""


class TTSModelManager:
    """Encapsulates loading and using a Coqui TTS model."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tts: Optional[TTS] = None
        self._model_path: Optional[Path] = None
        self._config_path: Optional[Path] = None
        self._speakers: List[str] = []
        self._languages: List[str] = []

    @property
    def speakers(self) -> List[str]:
        with self._lock:
            return list(self._speakers)

    @property
    def languages(self) -> List[str]:
        with self._lock:
            return list(self._languages)

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._tts is not None

    @property
    def model_path(self) -> Optional[Path]:
        with self._lock:
            return self._model_path

    def load_model(
        self,
        model_path: Path,
        config_path: Optional[Path] = None,
        *,
        progress_callback=None,
    ) -> None:
        """Load a model from disk.

        Args:
            model_path: Path to the ``.pth`` checkpoint.
            config_path: Optional path to the model's ``config.json``.
            progress_callback: Callable that receives textual updates.
        """
        with self._lock:
            self._emit(progress_callback, "Loading model…")
            try:
                tts = TTS(model_path=str(model_path), config_path=str(config_path) if config_path else None)
            except Exception as exc:  # pragma: no cover - defensive logging only
                raise TTSModelError("Failed to load model") from exc

            self._tts = tts
            self._model_path = model_path
            self._config_path = config_path
            self._speakers = getattr(tts, "speakers", None) or []
            self._languages = getattr(tts, "languages", None) or []
            self._emit(
                progress_callback,
                "Model loaded successfully. Speakers: {}".format(
                    ", ".join(self._speakers) if self._speakers else "<none>"
                ),
            )

    def synthesize(
        self,
        text: str,
        output_file: Path,
        *,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        progress_callback=None,
    ) -> None:
        """Generate audio from text."""
        with self._lock:
            if not self._tts:
                raise TTSModelError("Load a model before synthesizing audio.")

            if not text.strip():
                raise TTSModelError("Please enter some text to synthesize.")

            kwargs = {}
            if speaker:
                kwargs["speaker"] = speaker
            if language:
                kwargs["language"] = language

            self._emit(progress_callback, "Synthesizing audio…")
            try:
                self._tts.tts_to_file(text=text, file_path=str(output_file), **kwargs)
            except Exception as exc:  # pragma: no cover - defensive logging only
                raise TTSModelError("Failed to synthesize audio") from exc
            self._emit(progress_callback, f"Saved audio to {output_file}")

    @staticmethod
    def _emit(callback, message: str) -> None:
        if callback:
            callback(message)
