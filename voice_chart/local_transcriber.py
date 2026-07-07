from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel


class LocalWhisperTranscriber:
    """Runs Whisper locally via faster-whisper. No API key, no network calls
    (after the model has been downloaded once and cached)."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "ko",
        prompt: str | None = None,
    ) -> str:
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=prompt,
        )
        return "".join(segment.text for segment in segments).strip()
