"""http tts engine using openai-compatible speech api."""

import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore

from .config import DEFAULT_SEED, MAX_CHUNK_SIZE, SAMPLE_RATE, TTS_HTTP_TIMEOUT

# voice cache: (api_base, ref_audio_path, ref_text) -> voice_id
_voice_cache: dict[tuple[str, str, str], str] = {}


@dataclass
class HTTPTTSConfig:
    """configuration for http tts engine."""

    api_base: str = "http://localhost:8080/v1"
    model: str = ""
    speaker: str = "default"
    language: str = "en"
    batch_size: int = 1
    chunk_size: int = MAX_CHUNK_SIZE
    temperature: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int = DEFAULT_SEED

    # unused by http engine but accessed by pooling code
    compile_model: bool = False


def _get_json(url: str) -> dict:
    """GET url, return parsed json response."""
    try:
        with urllib.request.urlopen(url, timeout=TTS_HTTP_TIMEOUT) as resp:
            parsed: dict = json.loads(resp.read())
            return parsed
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        raise RuntimeError(f"http {e.code}: {error_body}") from e


def _post_json(url: str, body: dict) -> bytes:
    """send json post request, return response bytes."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=TTS_HTTP_TIMEOUT) as resp:
            response_bytes: bytes = resp.read()
            return response_bytes
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        raise RuntimeError(f"http {e.code}: {error_body}") from e


def _post_multipart(
    url: str, fields: dict[str, str], files: dict[str, tuple[str, bytes]]
) -> dict:
    """send multipart form post, return parsed json response."""
    boundary = "----autiobook-boundary-7d4a6d158c9b"
    body_parts = []

    for name, value in fields.items():
        body_parts.append(f"--{boundary}\r\n".encode())
        body_parts.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        body_parts.append(value.encode() + b"\r\n")

    for name, (filename, data) in files.items():
        body_parts.append(f"--{boundary}\r\n".encode())
        body_parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        body_parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
        body_parts.append(data + b"\r\n")

    body_parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(body_parts)

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TTS_HTTP_TIMEOUT) as resp:
            parsed: dict = json.loads(resp.read())
            return parsed
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        raise RuntimeError(f"http {e.code}: {error_body}") from e


def _wav_bytes_to_numpy(wav_data: bytes) -> tuple[np.ndarray, int]:
    """decode wav bytes to numpy array and sample rate."""
    audio, sr = sf.read(io.BytesIO(wav_data), dtype="float32")
    return audio, sr


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """encode numpy array to wav bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


class HTTPTTSEngine:
    """tts engine that calls an openai-compatible speech api."""

    def __init__(self, config: HTTPTTSConfig | None = None):
        self.config = config or HTTPTTSConfig()

    def _speech_url(self) -> str:
        return f"{self.config.api_base}/audio/speech"

    def _voices_url(self) -> str:
        return f"{self.config.api_base}/audio/voices"

    def list_voices(self) -> list[str]:
        """fetch available preset voices for the configured model."""
        url = self._voices_url()
        if self.config.model:
            url = f"{url}?model={self.config.model}"
        resp = _get_json(url)
        # response shape: {model_name: [voice, ...]} or {"voices": [...]}
        if isinstance(resp, dict):
            for v in resp.values():
                if isinstance(v, list):
                    return [str(x) for x in v]
        return []

    def _synthesize_one(
        self, text: str, voice: str = "", instruct: str = ""
    ) -> np.ndarray:
        """synthesize a single text string to audio."""
        body: dict[str, Any] = {
            "model": self.config.model,
            "input": text,
            "voice": voice or self.config.speaker,
            "language": self.config.language,
            "response_format": "wav",
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.top_k is not None:
            body["top_k"] = self.config.top_k
        if self.config.repetition_penalty is not None:
            body["repetition_penalty"] = self.config.repetition_penalty
        if self.config.seed > 0:
            body["seed"] = self.config.seed
        if instruct:
            body["instructions"] = instruct

        wav_data = _post_json(self._speech_url(), body)
        audio, _ = _wav_bytes_to_numpy(wav_data)
        return audio

    def synthesize(
        self, text: str | list[str], instruct: str = "", speaker: str | None = None
    ) -> tuple[np.ndarray | list[np.ndarray], int]:
        """synthesize speech from text."""
        voice = speaker or self.config.speaker
        if isinstance(text, str):
            return self._synthesize_one(text, voice, instruct), SAMPLE_RATE

        results = [self._synthesize_one(t, voice, instruct) for t in text]
        return results, SAMPLE_RATE

    def design_voice(self, text: str, instruct: str) -> tuple[np.ndarray, int]:
        """generate speech with voice design instruction."""
        return self._synthesize_one(text, instruct=instruct), SAMPLE_RATE

    def _get_or_create_voice(
        self, ref_audio: np.ndarray | tuple | str, ref_text: str
    ) -> str:
        """create a server-side voice from reference audio, with caching."""
        # normalize ref_audio to (audio_array, sample_rate)
        if isinstance(ref_audio, (str, Path)):
            audio_path = str(ref_audio)
            audio_data, audio_sr = sf.read(audio_path, dtype="float32")
        elif isinstance(ref_audio, tuple):
            audio_data, audio_sr = ref_audio
            audio_path = f"<array:{id(audio_data)}>"
        else:
            audio_data = ref_audio
            audio_sr = SAMPLE_RATE
            audio_path = f"<array:{id(audio_data)}>"

        cache_key = (self.config.api_base, audio_path, ref_text)
        if cache_key in _voice_cache:
            return _voice_cache[cache_key]

        wav_bytes = _numpy_to_wav_bytes(audio_data, audio_sr)

        fields = {"name": "autiobook_clone", "ref_text": ref_text}
        if self.config.model:
            fields["model"] = self.config.model

        resp = _post_multipart(
            self._voices_url(),
            fields=fields,
            files={"audio_sample": ("reference.wav", wav_bytes)},
        )

        voice_id: str = resp["id"]
        _voice_cache[cache_key] = voice_id
        print(f"created server voice: {voice_id} ({resp.get('mode', 'xvec')})")
        return voice_id

    def clone_voice(
        self,
        text: str | list[str],
        ref_audio: np.ndarray | tuple | str,
        ref_text: str,
    ) -> tuple[np.ndarray | list[np.ndarray], int]:
        """clone voice from reference audio via server api."""
        voice_id = self._get_or_create_voice(ref_audio, ref_text)

        if isinstance(text, str):
            return self._synthesize_one(text, voice=voice_id), SAMPLE_RATE

        results = [self._synthesize_one(t, voice=voice_id) for t in text]
        return results, SAMPLE_RATE
