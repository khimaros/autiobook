"""http tts engine using openai-compatible speech api."""

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore

from .config import (
    MAX_CHUNK_SIZE,
    SAMPLE_RATE,
    TTS_HTTP_TIMEOUT,
    TTS_STREAM_BATCH_SIZE,
    active_seed,
)

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
    seed: int = field(default_factory=active_seed)
    # when > 0, streaming audio calls request live per-batch PCM deltas so
    # playback can begin before synthesis finishes. set via
    # AUTIOBOOK_TTS_STREAM_BATCH_SIZE; 0 disables streaming.
    stream_batch_size: int = TTS_STREAM_BATCH_SIZE

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


def _post_sse(url: str, body: dict) -> tuple[bytes, dict, dict]:
    """POST json body, parse SSE response, return (audio_bytes, usage, timings).

    expects openai-compatible speech SSE: `speech.audio.delta` events carrying
    base64 audio fragments, terminated by a `speech.audio.done` event containing
    optional usage/timings metadata.
    """
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    audio_b64_parts: list[str] = []
    usage: dict = {}
    timings: dict = {}
    current_event: str | None = None
    data_buffer: list[str] = []

    def dispatch() -> None:
        nonlocal current_event, data_buffer, usage, timings
        if not data_buffer and current_event is None:
            return
        event_name = current_event or "message"
        payload = "".join(data_buffer)
        current_event = None
        data_buffer = []
        if not payload or payload == "[DONE]":
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return
        if event_name == "speech.audio.delta":
            audio = obj.get("audio")
            if isinstance(audio, str):
                audio_b64_parts.append(audio)
        elif event_name == "speech.audio.done":
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            if isinstance(obj.get("timings"), dict):
                timings = obj["timings"]

    try:
        with urllib.request.urlopen(req, timeout=TTS_HTTP_TIMEOUT) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line == "":
                    dispatch()
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    data_buffer.append(line[5:].lstrip())
            dispatch()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        raise RuntimeError(f"http {e.code}: {error_body}") from e

    if not audio_b64_parts:
        raise RuntimeError("tts sse response contained no audio")

    audio_bytes = base64.b64decode("".join(audio_b64_parts))
    return audio_bytes, usage, timings


def _post_sse_pcm_live(
    url: str,
    body: dict,
    on_chunk: Any | None = None,
    cancel: Any | None = None,
) -> tuple[bytes, dict, dict]:
    """POST json, parse SSE, call on_chunk(bytes) as each PCM delta arrives.

    server must emit raw PCM inside each speech.audio.delta (response_format=pcm
    and stream_batch_size>0). returns (full_pcm, usage, timings).

    if `cancel` is a threading.Event and it becomes set, stops reading and
    closes the connection early (partial pcm is still returned).
    """
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    pcm_parts: list[bytes] = []
    usage: dict = {}
    timings: dict = {}
    current_event: str | None = None
    data_buffer: list[str] = []

    def dispatch() -> None:
        nonlocal current_event, data_buffer, usage, timings
        if not data_buffer and current_event is None:
            return
        event_name = current_event or "message"
        payload = "".join(data_buffer)
        current_event = None
        data_buffer = []
        if not payload or payload == "[DONE]":
            return
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return
        if event_name == "speech.audio.delta":
            audio_b64 = obj.get("audio")
            if isinstance(audio_b64, str):
                chunk = base64.b64decode(audio_b64)
                pcm_parts.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
        elif event_name == "speech.audio.done":
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            if isinstance(obj.get("timings"), dict):
                timings = obj["timings"]

    try:
        with urllib.request.urlopen(req, timeout=TTS_HTTP_TIMEOUT) as resp:
            for raw in resp:
                if cancel is not None and cancel.is_set():
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line == "":
                    dispatch()
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    current_event = line[6:].strip()
                elif line.startswith("data:"):
                    data_buffer.append(line[5:].lstrip())
            if cancel is None or not cancel.is_set():
                dispatch()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        raise RuntimeError(f"http {e.code}: {error_body}") from e

    cancelled = cancel is not None and cancel.is_set()
    if not pcm_parts and not cancelled:
        raise RuntimeError("tts sse response contained no audio")

    return b"".join(pcm_parts), usage, timings


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

        # try SSE first so llama-swap can capture usage/timings and so we're
        # compatible with future progressive-playback servers. fall back to
        # non-streaming binary wav for servers that don't support stream_format.
        try:
            sse_body = {**body, "stream_format": "sse"}
            wav_data, _usage, _timings = _post_sse(self._speech_url(), sse_body)
        except RuntimeError:
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

    def design_voice_stream(
        self,
        text: str,
        instruct: str,
        on_chunk: Any | None = None,
        cancel: Any | None = None,
    ) -> tuple[np.ndarray, int]:
        """streaming variant of design_voice: calls on_chunk(pcm_bytes) as each
        batch arrives from the server. requires a server that supports
        stream_batch_size; raises RuntimeError on non-2xx.
        """
        batch_size = max(1, int(self.config.stream_batch_size or 16))
        body: dict[str, Any] = {
            "model": self.config.model,
            "input": text,
            "voice": self.config.speaker,
            "language": self.config.language,
            "response_format": "pcm",
            "stream_format": "sse",
            "stream_batch_size": batch_size,
            "instructions": instruct,
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.top_k is not None:
            body["top_k"] = self.config.top_k
        if self.config.repetition_penalty is not None:
            body["repetition_penalty"] = self.config.repetition_penalty
        if self.config.seed > 0:
            body["seed"] = self.config.seed

        pcm_bytes, _usage, _timings = _post_sse_pcm_live(
            self._speech_url(), body, on_chunk=on_chunk, cancel=cancel
        )
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, SAMPLE_RATE

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
