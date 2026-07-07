from __future__ import annotations

from pathlib import Path

from openai import OpenAI


class WhisperTranscriber:
    """Sends a recorded audio file to an OpenAI speech-to-text model."""

    def __init__(self, api_key: str | None = None, model: str = "whisper-1") -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "ko",
        prompt: str | None = None,
    ) -> str:
        with open(audio_path, "rb") as audio_file:
            response = self._client.audio.transcriptions.create(
                model=self._model,
                file=audio_file,
                language=language,
                prompt=prompt,
            )
        return response.text.strip()
