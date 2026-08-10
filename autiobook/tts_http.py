"""http tts engine using openai-compatible speech api."""

import base64
import http.client
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import soundfile as sf  # type: ignore
from tqdm import tqdm  # type: ignore

from .config import (
    DEFAULT_TTS_DIALECT,
    DEFAULT_TTS_DIRECTION,
    DEFAULT_TTS_RESPONSE_FORMAT,
    MAX_CHUNK_SIZE,
    OPENAI_DIALECT_HOSTS,
    SAMPLE_RATE,
    TTS_DIALECT_AUTO,
    TTS_DIALECT_OPENAI,
    TTS_DIALECT_QWEN,
    TTS_DIRECTION_PREFIX,
    TTS_DIRECTION_TEMPLATE,
    TTS_FORMAT_PCM,
    TTS_FORMAT_WAV,
    TTS_HTTP_TIMEOUT,
    TTS_MAX_RETRIES,
    TTS_PRESET_VOICES,
    TTS_RETRY_DELAY,
    TTS_RETRY_STATUS,
    TTS_STREAM_BATCH_SIZE,
    active_seed,
)
from .utils import retry_with_backoff

T = TypeVar("T")

# voice cache: (api_base, ref_audio_path, ref_text) -> voice_id
_voice_cache: dict[tuple[str, str, str], str] = {}


def _is_unknown_voice(err: Exception, voice_id: str) -> bool:
    """true if the server rejected voice_id as one it does not know."""
    msg = str(err).lower()
    return "unknown voice" in msg and voice_id.lower() in msg


PCM_DTYPE = np.int16
PCM_FULL_SCALE = 32768.0
# frames per delta when streaming is requested without a configured size
STREAM_BATCH_FALLBACK = 16
# bytes per socket read when streaming raw pcm; ~0.17s of audio at 24khz s16
STREAM_READ_BYTES = 8192

# voices published for hosted models that expose no discovery endpoint, keyed
# by a substring of the model id. openrouter has no voices api at all, so
# --preset-voices would otherwise need every name supplied by hand.
KNOWN_VOICES = {
    "gemini": [
        "Zephyr",
        "Puck",
        "Charon",
        "Kore",
        "Fenrir",
        "Leda",
        "Orus",
        "Aoede",
        "Callirrhoe",
        "Autonoe",
        "Enceladus",
        "Iapetus",
        "Umbriel",
        "Algieba",
        "Despina",
        "Erinome",
        "Algenib",
        "Rasalgethi",
        "Laomedeia",
        "Achernar",
        "Alnilam",
        "Schedar",
        "Gacrux",
        "Pulcherrima",
        "Achird",
        "Zubenelgenubi",
        "Vindemiatrix",
        "Sadachbia",
        "Sadaltager",
        "Sulafat",
    ],
}


def known_voices(model: str) -> list[str]:
    """built-in preset voice names for a hosted model id, or [] if unknown."""
    name = model.lower()
    for key, voices in KNOWN_VOICES.items():
        if key in name:
            return list(voices)
    return []


@dataclass
class HTTPTTSConfig:
    """configuration for http tts engine."""

    api_base: str = "http://localhost:8080/v1"
    api_key: str = ""
    # which request subset the backend understands; see config.TTS_DIALECTS
    dialect: str = DEFAULT_TTS_DIALECT
    # "" resolves from the dialect: wav locally, pcm on hosted providers
    response_format: str = DEFAULT_TTS_RESPONSE_FORMAT
    # how delivery direction is conveyed; see config.TTS_DIRECTIONS
    direction: str = DEFAULT_TTS_DIRECTION
    # preset voices for backends that expose no /audio/voices endpoint
    voices: list[str] = field(default_factory=lambda: list(TTS_PRESET_VOICES))
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


def resolve_dialect(dialect: str, api_base: str) -> str:
    """resolve `auto` to a concrete dialect from the endpoint host.

    hosted providers accept only the strict openai subset; anything else is
    assumed to be the local qwen3-tts server, whose api is a superset.
    """
    if dialect != TTS_DIALECT_AUTO:
        return dialect
    host = urllib.parse.urlparse(api_base).hostname or ""
    return TTS_DIALECT_OPENAI if host in OPENAI_DIALECT_HOSTS else TTS_DIALECT_QWEN


def resolve_response_format(response_format: str, dialect: str) -> str:
    """speech response format, defaulted to what the dialect's servers emit."""
    if response_format:
        return response_format
    return TTS_FORMAT_PCM if dialect == TTS_DIALECT_OPENAI else TTS_FORMAT_WAV


def _auth_headers(api_key: str) -> dict[str, str]:
    """bearer auth for hosted backends; empty for an unauthenticated server."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


# transport failures urllib hands back rather than wraps: reset or refused
# sockets and timeouts (OSError), plus http.client's own framing errors.
# RemoteDisconnected comes straight out of getresponse() unwrapped, which is
# why nothing downstream ever caught a provider dropping the connection.
TRANSPORT_ERRORS = (OSError, http.client.HTTPException)


def _retryable(err: Exception) -> bool:
    """true for a failure another attempt might get past."""
    if isinstance(err, urllib.error.HTTPError):
        return err.code in TTS_RETRY_STATUS
    return isinstance(err, TRANSPORT_ERRORS)


def _request(
    attempt: Callable[[], T], retryable: Callable[[Exception], bool] = _retryable
) -> T:
    """run a request with backoff, surfacing http errors with their body.

    the body is read only once the retries are spent: HTTPError.read() drains
    the response, and the status alone decides whether to send again.
    """
    try:
        return retry_with_backoff(
            attempt,
            TTS_MAX_RETRIES,
            TTS_RETRY_DELAY,
            retryable=retryable,
            report=tqdm.write,
            label="tts error",
        )
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        raise RuntimeError(f"http {e.code}: {error_body}") from e


def _json_request(
    url: str, body: dict, api_key: str, sse: bool = False
) -> urllib.request.Request:
    """a POST carrying a json body, optionally asking for an sse reply."""
    headers = {"Content-Type": "application/json", **_auth_headers(api_key)}
    if sse:
        headers["Accept"] = "text/event-stream"
    return urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)


def _get_json(url: str, api_key: str = "") -> dict:
    """GET url, return parsed json response."""
    req = urllib.request.Request(url, headers=_auth_headers(api_key))

    def attempt() -> dict:
        with urllib.request.urlopen(req, timeout=TTS_HTTP_TIMEOUT) as resp:
            parsed: dict = json.loads(resp.read())
            return parsed

    return _request(attempt)


def _read_sse(
    resp: Any, on_chunk: Any | None = None, cancel: Any | None = None
) -> tuple[bytes, dict, dict]:
    """parse an openai-compatible speech SSE stream off an open response.

    `speech.audio.delta` events carry base64 audio fragments, handed to
    on_chunk as they land, and a closing `speech.audio.done` carries optional
    usage/timings. each delta is decoded on its own: they are separately
    encoded, so joining the base64 first would trip over embedded padding.

    if `cancel` is a threading.Event and it becomes set, stops reading and
    closes the connection early (partial audio is still returned).
    """
    parts: list[bytes] = []
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
                chunk = base64.b64decode(audio)
                parts.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
        elif event_name == "speech.audio.done":
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            if isinstance(obj.get("timings"), dict):
                timings = obj["timings"]

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

    return b"".join(parts), usage, timings


def _post_sse(
    url: str,
    body: dict,
    on_chunk: Any | None = None,
    cancel: Any | None = None,
    api_key: str = "",
) -> tuple[bytes, dict, dict]:
    """POST json body, parse the SSE reply, return (audio, usage, timings).

    an on_chunk callback receives each delta as it arrives, so playback can
    begin before synthesis finishes; that needs the server emitting raw pcm
    (response_format=pcm, stream_batch_size>0). a failure after any of the
    take has been handed over is final -- a retry would replay its opening.
    """
    req = _json_request(url, body, api_key, sse=True)
    delivered = False

    def relay(chunk: bytes) -> None:
        nonlocal delivered
        delivered = True
        if on_chunk is not None:
            on_chunk(chunk)

    def attempt() -> tuple[bytes, dict, dict]:
        with urllib.request.urlopen(req, timeout=TTS_HTTP_TIMEOUT) as resp:
            audio, usage, timings = _read_sse(
                resp, on_chunk=relay if on_chunk else None, cancel=cancel
            )
        cancelled = cancel is not None and cancel.is_set()
        if not audio and not cancelled:
            raise RuntimeError("tts sse response contained no audio")
        return audio, usage, timings

    return _request(attempt, lambda e: not delivered and _retryable(e))


def _post_stream(
    url: str,
    body: dict,
    on_chunk: Any | None = None,
    cancel: Any | None = None,
    api_key: str = "",
) -> bytes:
    """POST json, call on_chunk(bytes) as the audio body arrives, return it all.

    hosted providers answer /audio/speech with a chunked byte stream rather
    than SSE, so progressive playback means reading the socket incrementally.
    read1 hands back whatever has landed instead of waiting for a full buffer.

    if `cancel` is a threading.Event and it becomes set, stops reading early
    (partial audio is still returned). as with _post_sse, audio already handed
    to on_chunk rules out another attempt.
    """
    req = _json_request(url, body, api_key)
    delivered = False

    def attempt() -> bytes:
        nonlocal delivered
        parts: list[bytes] = []
        with urllib.request.urlopen(req, timeout=TTS_HTTP_TIMEOUT) as resp:
            read = getattr(resp, "read1", resp.read)
            while cancel is None or not cancel.is_set():
                chunk = read(STREAM_READ_BYTES)
                if not chunk:
                    break
                parts.append(chunk)
                if on_chunk is not None:
                    delivered = True
                    on_chunk(chunk)
        if not parts and (cancel is None or not cancel.is_set()):
            raise RuntimeError("tts response contained no audio")
        return b"".join(parts)

    return _request(attempt, lambda e: not delivered and _retryable(e))


def _post_json(url: str, body: dict, api_key: str = "") -> bytes:
    """send json post request, return response bytes."""
    req = _json_request(url, body, api_key)

    def attempt() -> bytes:
        with urllib.request.urlopen(req, timeout=TTS_HTTP_TIMEOUT) as resp:
            response_bytes: bytes = resp.read()
            return response_bytes

    return _request(attempt)


def _post_multipart(
    url: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes]],
    api_key: str = "",
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
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            **_auth_headers(api_key),
        },
    )

    def attempt() -> dict:
        with urllib.request.urlopen(req, timeout=TTS_HTTP_TIMEOUT) as resp:
            parsed: dict = json.loads(resp.read())
            return parsed

    return _request(attempt)


def _pcm_bytes_to_numpy(pcm_data: bytes) -> np.ndarray:
    """decode signed 16-bit little-endian pcm to float32 samples."""
    return np.frombuffer(pcm_data, dtype=PCM_DTYPE).astype(np.float32) / PCM_FULL_SCALE


def _wav_bytes_to_numpy(wav_data: bytes) -> tuple[np.ndarray, int]:
    """decode container bytes (wav, mp3) to numpy array and sample rate."""
    audio, sr = sf.read(io.BytesIO(wav_data), dtype="float32")
    return audio, sr


def _decode_audio(data: bytes, response_format: str) -> np.ndarray:
    """decode a speech response body into float32 samples.

    raw pcm carries no header, so its rate is assumed to be SAMPLE_RATE --
    what openrouter, openai and the local server all emit.
    """
    if response_format == TTS_FORMAT_PCM:
        return _pcm_bytes_to_numpy(data)
    audio, _ = _wav_bytes_to_numpy(data)
    return audio


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """encode numpy array to wav bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


class HTTPTTSEngine:
    """tts engine that calls an openai-compatible speech api."""

    def __init__(self, config: HTTPTTSConfig | None = None):
        self.config = config or HTTPTTSConfig()

    @property
    def dialect(self) -> str:
        """request subset this backend understands (see config.TTS_DIALECTS)."""
        return resolve_dialect(self.config.dialect, self.config.api_base)

    @property
    def response_format(self) -> str:
        return resolve_response_format(self.config.response_format, self.dialect)

    @property
    def seeded(self) -> bool:
        """whether the backend varies output with the seed field.

        hosted providers document no seed on /audio/speech, so retries there
        explore fresh samples but cannot be reproduced.
        """
        return self.dialect != TTS_DIALECT_OPENAI

    @property
    def streaming(self) -> bool:
        """whether takes can play while they render.

        the qwen server streams only when asked for a batch size; a hosted
        provider answers every request as a byte stream, so it always can.
        """
        if self.dialect == TTS_DIALECT_OPENAI:
            return True
        return self.config.stream_batch_size > 0

    def _speech_url(self) -> str:
        return f"{self.config.api_base}/audio/speech"

    def _voices_url(self) -> str:
        return f"{self.config.api_base}/audio/voices"

    def _require_qwen(self, feature: str) -> None:
        """reject a qwen-server-only feature on a hosted openai backend.

        hosted cloning exists but is a different mechanism (openrouter's
        stateless input_references, sent per request) that this engine does
        not speak yet, so preset voices are the supported route.
        """
        if self.dialect == TTS_DIALECT_OPENAI:
            raise RuntimeError(
                f"{feature} needs the qwen tts server; {self.config.api_base} "
                "serves preset voices only (run audition with --preset-voices)"
            )

    def list_voices(self) -> list[str]:
        """available preset voices for the configured model.

        hosted providers publish no discovery endpoint, so their voices come
        from configuration or the built-in table.
        """
        configured = self.config.voices or known_voices(self.config.model)
        if configured:
            return list(configured)
        if self.dialect == TTS_DIALECT_OPENAI:
            raise RuntimeError(
                f"{self.config.api_base} has no /audio/voices endpoint and no "
                f"voices are known for '{self.config.model}'; pass --tts-voices "
                "(or set AUTIOBOOK_TTS_VOICES) with the names it accepts"
            )
        url = self._voices_url()
        if self.config.model:
            url = f"{url}?model={self.config.model}"
        resp = _get_json(url, api_key=self.config.api_key)
        # response shape: {model_name: [voice, ...]} or {"voices": [...]}
        if isinstance(resp, dict):
            for v in resp.values():
                if isinstance(v, list):
                    return [str(x) for x in v]
        return []

    def _speech_body(
        self, text: str, voice: str = "", instruct: str = ""
    ) -> dict[str, Any]:
        """build the /audio/speech body accepted by the configured dialect.

        the qwen sampler and language fields are omitted for hosted providers:
        they are silently dropped at best and rejected at worst, and a rejected
        request is still a billable round trip.
        """
        if instruct and self.config.direction == TTS_DIRECTION_PREFIX:
            text = TTS_DIRECTION_TEMPLATE.format(instruct=instruct, text=text)
            instruct = ""

        body: dict[str, Any] = {
            "model": self.config.model,
            "input": text,
            "voice": voice or self.config.speaker,
            "response_format": self.response_format,
        }
        if instruct:
            body["instructions"] = instruct
        if self.dialect == TTS_DIALECT_OPENAI:
            return body

        body["language"] = self.config.language
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.top_k is not None:
            body["top_k"] = self.config.top_k
        if self.config.repetition_penalty is not None:
            body["repetition_penalty"] = self.config.repetition_penalty
        if self.config.seed > 0:
            body["seed"] = self.config.seed
        return body

    def _synthesize_one(
        self, text: str, voice: str = "", instruct: str = ""
    ) -> np.ndarray:
        """synthesize a single text string to audio."""
        body = self._speech_body(text, voice, instruct)
        url = self._speech_url()
        key = self.config.api_key

        # the qwen server is asked over SSE first so llama-swap can record
        # usage/timings. hosted providers answer with raw bytes, so probing
        # them just buys a second billed synthesis of the same text.
        if self.dialect == TTS_DIALECT_QWEN:
            try:
                data, _usage, _timings = _post_sse(
                    url, {**body, "stream_format": "sse"}, api_key=key
                )
            except (RuntimeError, *TRANSPORT_ERRORS):
                # a server whose sse endpoint is broken or gone may still
                # answer the plain post, and it costs one more local request
                data = _post_json(url, body, api_key=key)
        else:
            data = _post_json(url, body, api_key=key)
        return _decode_audio(data, self.response_format)

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
        voice: str = "",
    ) -> tuple[np.ndarray, int]:
        """streaming variant of design_voice: calls on_chunk(pcm_bytes) as audio
        arrives, so playback can begin before synthesis finishes.

        pcm is forced regardless of the configured response format: the player
        is fed raw samples, and a container would arrive header-first.
        """
        body = {
            **self._speech_body(text, voice, instruct),
            "response_format": TTS_FORMAT_PCM,
        }
        url = self._speech_url()
        key = self.config.api_key

        # the qwen server emits pcm inside sse deltas and needs a batch size;
        # hosted providers just stream the response body.
        if self.dialect == TTS_DIALECT_QWEN:
            batch_size = max(
                1, int(self.config.stream_batch_size or STREAM_BATCH_FALLBACK)
            )
            body["stream_format"] = "sse"
            body["stream_batch_size"] = batch_size
            pcm_bytes, _usage, _timings = _post_sse(
                url, body, on_chunk=on_chunk, cancel=cancel, api_key=key
            )
        else:
            pcm_bytes = _post_stream(
                url, body, on_chunk=on_chunk, cancel=cancel, api_key=key
            )
        return _pcm_bytes_to_numpy(pcm_bytes), SAMPLE_RATE

    def _get_or_create_voice(
        self, ref_audio: np.ndarray | tuple | str, ref_text: str, refresh: bool = False
    ) -> str:
        """create a server-side voice from reference audio, with caching.

        refresh=True re-registers even on a cache hit, for when the server has
        forgotten the id we hold."""
        self._require_qwen("voice cloning")
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
        if not refresh and cache_key in _voice_cache:
            return _voice_cache[cache_key]

        wav_bytes = _numpy_to_wav_bytes(audio_data, audio_sr)

        fields = {"name": "autiobook_clone", "ref_text": ref_text}
        if self.config.model:
            fields["model"] = self.config.model

        resp = _post_multipart(
            self._voices_url(),
            fields=fields,
            files={"audio_sample": ("reference.wav", wav_bytes)},
            api_key=self.config.api_key,
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
        """clone voice from reference audio via server api.

        server voice ids live in the model process's memory, so a swap or
        restart invalidates every id we hold. re-register once and retry rather
        than failing every remaining batch of the run.
        """
        voice_id = self._get_or_create_voice(ref_audio, ref_text)
        texts = [text] if isinstance(text, str) else list(text)

        try:
            results = [self._synthesize_one(t, voice=voice_id) for t in texts]
        except RuntimeError as e:
            if not _is_unknown_voice(e, voice_id):
                raise
            print(f"server forgot voice {voice_id}; re-registering...")
            voice_id = self._get_or_create_voice(ref_audio, ref_text, refresh=True)
            results = [self._synthesize_one(t, voice=voice_id) for t in texts]

        if isinstance(text, str):
            return results[0], SAMPLE_RATE
        return results, SAMPLE_RATE
