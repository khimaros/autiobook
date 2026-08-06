"""tests for the http tts engine across local and hosted backends."""

import argparse
import io
import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf  # type: ignore

from autiobook.config import (
    SAMPLE_RATE,
    TTS_DIALECT_OPENAI,
    TTS_DIALECT_QWEN,
    TTS_DIRECTION_PREFIX,
    TTS_FORMAT_MP3,
    TTS_FORMAT_PCM,
    TTS_FORMAT_WAV,
)
from autiobook.tts_http import HTTPTTSConfig, HTTPTTSEngine

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
LOCAL_BASE = "http://localhost:8080/v1"
# comfortably more than one STREAM_READ_BYTES buffer of s16 samples
LONG_TAKE_SAMPLES = 20000


def _wav_bytes(seconds: float = 0.01) -> bytes:
    """a real wav payload so decoding is exercised, not mocked."""
    buf = io.BytesIO()
    samples = np.zeros(int(SAMPLE_RATE * seconds), dtype="float32")
    sf.write(buf, samples, SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def _pcm_bytes(count: int = 8) -> bytes:
    return np.arange(count, dtype=np.int16).tobytes()


def _mock_response(payload: bytes):
    resp = MagicMock()
    resp.read.return_value = payload
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    # SSE parsing iterates the response line by line
    resp.__iter__ = lambda s: iter(payload.splitlines(keepends=True))
    return resp


def _engine(**kwargs) -> HTTPTTSEngine:
    return HTTPTTSEngine(HTTPTTSConfig(model="tts-1", **kwargs))


def _bodies(mock_url) -> list[dict]:
    return [json.loads(c[0][0].data) for c in mock_url.call_args_list]


class TestDialectResolution:
    """auto-detection of the request subset a backend understands."""

    def test_openrouter_host_resolves_openai(self):
        assert _engine(api_base=OPENROUTER_BASE).dialect == TTS_DIALECT_OPENAI

    def test_openai_host_resolves_openai(self):
        engine = _engine(api_base="https://api.openai.com/v1")
        assert engine.dialect == TTS_DIALECT_OPENAI

    def test_local_host_resolves_qwen(self):
        assert _engine(api_base=LOCAL_BASE).dialect == TTS_DIALECT_QWEN

    def test_explicit_dialect_wins_over_host(self):
        engine = _engine(api_base=OPENROUTER_BASE, dialect=TTS_DIALECT_QWEN)
        assert engine.dialect == TTS_DIALECT_QWEN

    def test_response_format_follows_dialect(self):
        assert _engine(api_base=OPENROUTER_BASE).response_format == TTS_FORMAT_PCM
        assert _engine(api_base=LOCAL_BASE).response_format == TTS_FORMAT_WAV

    def test_explicit_response_format_wins(self):
        engine = _engine(api_base=OPENROUTER_BASE, response_format=TTS_FORMAT_MP3)
        assert engine.response_format == TTS_FORMAT_MP3


class TestAuth:
    """bearer auth reaches every tts request, not just the llm ones."""

    def test_api_key_sent_on_speech(self):
        engine = _engine(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes())
        ) as mock_url:
            engine.synthesize("hello")
            req = mock_url.call_args[0][0]
            assert req.get_header("Authorization") == "Bearer sk-or-test"

    def test_no_key_sends_no_auth_header(self):
        engine = _engine(api_base=LOCAL_BASE)
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_wav_bytes())
        ) as mock_url:
            engine.synthesize("hello")
            req = mock_url.call_args[0][0]
            assert req.get_header("Authorization") is None

    def test_api_key_sent_on_voices_listing(self):
        engine = _engine(api_base=LOCAL_BASE, api_key="sk-test")
        payload = json.dumps({"tts-1": ["ryan"]}).encode()
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(payload)
        ) as mock_url:
            assert engine.list_voices() == ["ryan"]
            req = mock_url.call_args[0][0]
            assert req.get_header("Authorization") == "Bearer sk-test"


class TestOpenAIDialectRequests:
    """hosted providers accept a narrow body and bill every round trip."""

    def test_body_limited_to_supported_fields(self):
        engine = _engine(
            api_base=OPENROUTER_BASE,
            api_key="sk-or-test",
            speaker="Zephyr",
            temperature=0.7,
            top_k=20,
            repetition_penalty=1.1,
            seed=1234,
        )
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes())
        ) as mock_url:
            engine.synthesize("hello")
            body = _bodies(mock_url)[0]

        assert body == {
            "model": "tts-1",
            "input": "hello",
            "voice": "Zephyr",
            "response_format": TTS_FORMAT_PCM,
        }

    def test_single_request_no_sse_probe(self):
        """the qwen sse probe would be a second billed synthesis."""
        engine = _engine(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes())
        ) as mock_url:
            engine.synthesize("hello")

        assert mock_url.call_count == 1
        assert "stream_format" not in _bodies(mock_url)[0]

    def test_pcm_response_decoded(self):
        engine = _engine(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes(4))
        ):
            audio, sr = engine.synthesize("hello")

        assert sr == SAMPLE_RATE
        assert audio.dtype == np.float32
        np.testing.assert_allclose(
            audio, np.arange(4, dtype=np.float32) / 32768.0, rtol=1e-6
        )

    def test_instructions_sent_as_field_by_default(self):
        engine = _engine(api_base=OPENROUTER_BASE)
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes())
        ) as mock_url:
            engine.synthesize("hello", instruct="speaks joyfully")
            body = _bodies(mock_url)[0]

        assert body["instructions"] == "speaks joyfully"
        assert body["input"] == "hello"

    def test_prefix_direction_folds_into_input(self):
        """providers that drop unknown fields only take direction inline."""
        engine = _engine(api_base=OPENROUTER_BASE, direction=TTS_DIRECTION_PREFIX)
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_pcm_bytes())
        ) as mock_url:
            engine.synthesize("hello", instruct="speaks joyfully")
            body = _bodies(mock_url)[0]

        assert "instructions" not in body
        assert "speaks joyfully" in body["input"]
        assert body["input"].endswith("hello")


class TestQwenDialectRequests:
    """the local server keeps its sse probe and sampler fields."""

    def test_sse_probe_first_then_falls_back(self):
        """a server that cannot stream still answers the plain wav retry."""
        engine = _engine(api_base=LOCAL_BASE)
        sse_bodies: list[dict] = []

        def record_sse(url, body, api_key=""):
            sse_bodies.append(body)
            raise RuntimeError("no sse")

        with patch("autiobook.tts_http._post_sse", side_effect=record_sse):
            with patch(
                "urllib.request.urlopen", return_value=_mock_response(_wav_bytes())
            ) as mock_url:
                engine.synthesize("hello")

        assert sse_bodies[0]["stream_format"] == "sse"
        assert "stream_format" not in _bodies(mock_url)[0]

    def test_sampler_fields_sent(self):
        engine = _engine(
            api_base=LOCAL_BASE,
            temperature=0.7,
            top_k=20,
            repetition_penalty=1.1,
            seed=99,
        )
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(_wav_bytes())
        ) as mock_url:
            with patch(
                "autiobook.tts_http._post_sse", side_effect=RuntimeError("no sse")
            ):
                engine.synthesize("hello")
            body = _bodies(mock_url)[0]

        assert body["temperature"] == 0.7
        assert body["top_k"] == 20
        assert body["repetition_penalty"] == 1.1
        assert body["seed"] == 99
        assert body["language"] == "en"
        assert body["response_format"] == TTS_FORMAT_WAV


class TestPresetVoices:
    """--preset-voices needs a voice list even without a discovery endpoint."""

    def test_known_model_voices_without_request(self):
        engine = _engine(api_base=OPENROUTER_BASE)
        engine.config.model = "google/gemini-3.1-flash-tts-preview"
        with patch("urllib.request.urlopen") as mock_url:
            voices = engine.list_voices()

        mock_url.assert_not_called()
        assert voices[0] == "Zephyr"
        assert "Sulafat" in voices

    def test_configured_voices_override_known(self):
        engine = _engine(api_base=OPENROUTER_BASE, voices=["Puck", "Kore"])
        engine.config.model = "google/gemini-3.1-flash-tts-preview"
        assert engine.list_voices() == ["Puck", "Kore"]

    def test_unknown_hosted_model_reports_how_to_fix(self):
        engine = _engine(api_base=OPENROUTER_BASE)
        engine.config.model = "some/unlisted-tts"
        with pytest.raises(RuntimeError, match="--tts-voices"):
            engine.list_voices()

    def test_qwen_still_queries_the_endpoint(self):
        engine = _engine(api_base=LOCAL_BASE)
        payload = json.dumps({"tts-1": ["ryan", "vivian"]}).encode()
        with patch(
            "urllib.request.urlopen", return_value=_mock_response(payload)
        ) as mock_url:
            assert engine.list_voices() == ["ryan", "vivian"]
            assert "/audio/voices" in mock_url.call_args[0][0].full_url


class TestUnsupportedFeatures:
    """hosted backends have no server-side voice creation."""

    def test_clone_voice_explains_the_alternative(self):
        engine = _engine(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        with pytest.raises(RuntimeError, match="--preset-voices"):
            engine.clone_voice("hello", np.zeros(10, dtype="float32"), "ref")


class TestStreaming:
    """progressive playback, by sse locally and chunked reads when hosted."""

    def test_hosted_backends_always_stream(self):
        """no batch size to configure: the response body is the stream."""
        assert _engine(api_base=OPENROUTER_BASE).streaming is True

    def test_qwen_streams_only_when_asked(self):
        assert _engine(api_base=LOCAL_BASE).streaming is False
        assert _engine(api_base=LOCAL_BASE, stream_batch_size=16).streaming is True

    def test_hosted_stream_reads_body_incrementally(self, stub_provider):
        engine = HTTPTTSEngine(
            HTTPTTSConfig(
                api_base=stub_provider,
                api_key="sk-or-test",
                dialect=TTS_DIALECT_OPENAI,
                model="tts-1",
            )
        )
        chunks: list[bytes] = []
        audio, sr = engine.design_voice_stream(
            "long", "warm voice", on_chunk=chunks.append, voice="Puck"
        )

        # the point of streaming: audio is handed over before the take ends
        assert len(chunks) > 1
        assert b"".join(chunks) == _pcm_bytes(LONG_TAKE_SAMPLES)
        assert len(audio) == LONG_TAKE_SAMPLES
        assert sr == SAMPLE_RATE
        sent = StubProvider.requests[0]["body"]
        assert sent["voice"] == "Puck"
        assert sent["response_format"] == TTS_FORMAT_PCM
        assert "stream_format" not in sent

    def test_hosted_stream_forces_pcm_over_configured_format(self, stub_provider):
        """the player is fed raw samples; a container would arrive header-first."""
        engine = HTTPTTSEngine(
            HTTPTTSConfig(
                api_base=stub_provider,
                api_key="sk-or-test",
                dialect=TTS_DIALECT_OPENAI,
                response_format=TTS_FORMAT_MP3,
                model="tts-1",
            )
        )
        engine.design_voice_stream("hello", "warm voice")
        assert StubProvider.requests[0]["body"]["response_format"] == TTS_FORMAT_PCM

    def test_qwen_still_uses_sse(self):
        engine = _engine(api_base=LOCAL_BASE, stream_batch_size=8)
        with patch(
            "autiobook.tts_http._post_sse_pcm_live",
            return_value=(_pcm_bytes(2), {}, {}),
        ) as mock_sse:
            engine.design_voice_stream("hello", "warm voice")

        body = mock_sse.call_args[0][1]
        assert body["stream_format"] == "sse"
        assert body["stream_batch_size"] == 8


class StubProvider(BaseHTTPRequestHandler):
    """minimal hosted tts backend: bearer auth, raw pcm, no voices endpoint."""

    requests: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubProvider.requests.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": body,
            }
        )
        if body.get("response_format") != TTS_FORMAT_PCM:
            self.send_error(400, "unsupported response_format")
            return
        # "long" asks for more audio than one socket read returns, so a
        # streaming client observably receives it in pieces
        payload = _pcm_bytes(LONG_TAKE_SAMPLES if body["input"] == "long" else 4)
        self.send_response(200)
        self.send_header("Content-Type", "audio/pcm")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.send_error(404, "not found")

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_provider():
    """a real http server, so headers and framing are exercised end to end."""
    StubProvider.requests = []
    server = HTTPServer(("127.0.0.1", 0), StubProvider)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}/v1"
    server.shutdown()
    server.server_close()


class TestAgainstStubProvider:
    """full request/response cycle over a socket, no mocks."""

    def _engine(self, base: str) -> HTTPTTSEngine:
        return HTTPTTSEngine(
            HTTPTTSConfig(
                api_base=base,
                api_key="sk-or-test",
                dialect=TTS_DIALECT_OPENAI,
                model="google/gemini-3.1-flash-tts-preview",
                speaker="Zephyr",
            )
        )

    def test_synthesis_round_trip(self, stub_provider):
        audio, sr = self._engine(stub_provider).synthesize("hello")

        assert sr == SAMPLE_RATE
        assert len(audio) == 4
        sent = StubProvider.requests[0]
        assert sent["path"] == "/v1/audio/speech"
        assert sent["auth"] == "Bearer sk-or-test"
        assert sent["body"]["voice"] == "Zephyr"

    def test_one_request_per_chunk(self, stub_provider):
        self._engine(stub_provider).synthesize(["one", "two"])
        assert len(StubProvider.requests) == 2

    def test_voices_resolve_without_an_endpoint(self, stub_provider):
        """the stub 404s /audio/voices, as openrouter does."""
        voices = self._engine(stub_provider).list_voices()
        assert "Zephyr" in voices
        assert StubProvider.requests == []


class TestConfigFromArgs:
    """cli/env wiring reaches the engine."""

    def _args(self, **kwargs):
        defaults = dict(
            api_base=None,
            api_key=None,
            tts_api_base=None,
            tts_api_key=None,
            tts_dialect="auto",
            tts_direction="field",
            tts_voices="",
            tts_model="",
        )
        return argparse.Namespace(**{**defaults, **kwargs})

    def test_api_key_reaches_tts_config(self):
        from autiobook.utils import get_tts_config

        config = get_tts_config(
            self._args(api_base=OPENROUTER_BASE, api_key="sk-or-test")
        )
        assert config.api_base == OPENROUTER_BASE
        assert config.api_key == "sk-or-test"

    def test_tts_endpoint_overrides_shared_one(self):
        from autiobook.utils import get_tts_config

        config = get_tts_config(
            self._args(
                api_base=LOCAL_BASE,
                api_key="sk-llm",
                tts_api_base=OPENROUTER_BASE,
                tts_api_key="sk-or-test",
            )
        )
        assert config.api_base == OPENROUTER_BASE
        assert config.api_key == "sk-or-test"

    def test_tts_endpoint_alone_selects_http_engine(self):
        from autiobook.tts_http import HTTPTTSConfig as Cfg
        from autiobook.utils import get_tts_config

        config = get_tts_config(self._args(tts_api_base=OPENROUTER_BASE))
        assert isinstance(config, Cfg)

    def test_voices_flag_parsed(self):
        from autiobook.utils import get_design_config

        config = get_design_config(
            self._args(api_base=OPENROUTER_BASE, tts_voices="Zephyr, Puck ,Kore")
        )
        assert config.voices == ["Zephyr", "Puck", "Kore"]

    def test_tts_model_overrides_every_mode(self):
        """one hosted model serves design and cloning; the qwen ids do not."""
        from autiobook.utils import get_clone_config, get_design_config, get_tts_config

        args = self._args(api_base=OPENROUTER_BASE, tts_model="google/gemini-tts")
        assert get_tts_config(args).model == "google/gemini-tts"
        assert get_design_config(args).model == "google/gemini-tts"
        assert get_clone_config(args).model == "google/gemini-tts"

    def test_per_mode_model_still_wins(self):
        from autiobook.utils import get_clone_config

        args = self._args(
            api_base=OPENROUTER_BASE,
            tts_model="google/gemini-tts",
            tts_clone_model="other/clone-tts",
        )
        assert get_clone_config(args).model == "other/clone-tts"


class TestPresetAuditionConfig:
    """dramatize's preset path must not rebuild the config from scratch."""

    def _design_config(self) -> HTTPTTSConfig:
        return HTTPTTSConfig(
            api_base=OPENROUTER_BASE,
            api_key="sk-or-test",
            model="google/gemini-3.1-flash-tts-preview",
            direction=TTS_DIRECTION_PREFIX,
            voices=["Zephyr"],
        )

    def test_hosted_config_keeps_auth_and_model(self):
        from autiobook.utils import preset_audition_config

        config = preset_audition_config(self._design_config())

        assert config.api_key == "sk-or-test"
        assert config.model == "google/gemini-3.1-flash-tts-preview"
        assert config.direction == TTS_DIRECTION_PREFIX
        assert config.voices == ["Zephyr"]

    def test_qwen_config_switches_to_the_instructable_model(self):
        from autiobook.config import DEFAULT_MODEL
        from autiobook.utils import preset_audition_config

        design = replace(
            self._design_config(), api_base=LOCAL_BASE, model="design-model"
        )
        assert preset_audition_config(design).model == DEFAULT_MODEL

    def test_local_config_falls_back_to_a_fresh_http_config(self):
        """a local engine config has no endpoint to inherit."""
        from autiobook.utils import preset_audition_config

        config = preset_audition_config(object(), LOCAL_BASE)
        assert isinstance(config, HTTPTTSConfig)
        assert config.api_base == LOCAL_BASE
