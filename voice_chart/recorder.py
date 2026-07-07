from __future__ import annotations

import numpy as np
import sounddevice as sd
import soundfile as sf


class MicRecorder:
    """Records audio from the default microphone until stop() is called."""

    def __init__(self, samplerate: int = 16000, channels: int = 1) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time, status) -> None:
        if status:
            print(f"[recorder] {status}")
        self._frames.append(indata.copy())

    def start(self) -> None:
        self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("recording was not started")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._frames:
            return np.zeros((0, self.channels), dtype="float32")
        return np.concatenate(self._frames, axis=0)

    def save_wav(self, audio: np.ndarray, path: str) -> None:
        sf.write(path, audio, self.samplerate)
