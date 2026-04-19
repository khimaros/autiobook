"""tests for inline callback quality guard used during introduce/audition.

we patch `categorize_audio` to avoid brittle signal crafting — intent of
these tests is the retry/save/save logic, not the DSP heuristics themselves
(those live in tests/test_retake.py).
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autiobook.callback import (
    CALLBACK_MAX_ATTEMPTS,
    generate_with_callback,
)
from autiobook.config import SAMPLE_RATE, WAV_EXT
from autiobook.dramatize import run_auditions
from autiobook.introduce import run_introduce

# sentinel arrays: first sample tags "good" vs "bad" for _fake_categorize
_CLEAN = np.full(SAMPLE_RATE, 0.1, dtype=np.float32)
_BAD = np.zeros(SAMPLE_RATE, dtype=np.float32)


def _clean(_n: int = SAMPLE_RATE) -> np.ndarray:
    return _CLEAN.copy()


def _bad(_n: int = SAMPLE_RATE) -> np.ndarray:
    return _BAD.copy()


def _fake_categorize(audio, sr=SAMPLE_RATE):
    return [] if float(audio[0]) != 0.0 else ["silent"]


class TestGenerateWithCallback:
    def test_accepts_clean_audio_on_first_try(self):
        engine = SimpleNamespace(config=SimpleNamespace(seed=42))
        calls = []

        def gen():
            calls.append(engine.config.seed)
            return _clean(), SAMPLE_RATE

        with patch("autiobook.callback.categorize_audio", _fake_categorize):
            audio, sr = generate_with_callback(gen, engine, label="test")
        assert len(calls) == 1
        assert engine.config.seed == 42
        assert sr == SAMPLE_RATE

    def test_retries_and_bumps_seed(self):
        engine = SimpleNamespace(config=SimpleNamespace(seed=100))
        results = [
            (_bad(), SAMPLE_RATE),
            (_bad(), SAMPLE_RATE),
            (_clean(), SAMPLE_RATE),
        ]
        calls = []

        def gen():
            calls.append(engine.config.seed)
            return results.pop(0)

        with patch("autiobook.callback.categorize_audio", _fake_categorize):
            generate_with_callback(gen, engine, label="x")
        assert calls == [100, 101, 102]

    def test_does_not_bump_when_seed_disabled(self):
        engine = SimpleNamespace(config=SimpleNamespace(seed=0))
        results = [(_bad(), SAMPLE_RATE), (_clean(), SAMPLE_RATE)]

        def gen():
            return results.pop(0)

        with patch("autiobook.callback.categorize_audio", _fake_categorize):
            generate_with_callback(gen, engine, label="x")
        assert engine.config.seed == 0

    def test_raises_after_max_attempts(self):
        engine = SimpleNamespace(config=SimpleNamespace(seed=1))

        def gen():
            return _bad(), SAMPLE_RATE

        with patch("autiobook.callback.categorize_audio", _fake_categorize):
            with pytest.raises(RuntimeError, match="failed audio quality"):
                generate_with_callback(gen, engine, label="bad", max_attempts=3)


@pytest.fixture
def audition_workdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        cast_dir = workdir / "cast"
        cast_dir.mkdir()
        cast_data = {
            "version": 4,
            "characters": [
                {
                    "name": "Alice",
                    "description": "warm female voice",
                    "audition_line": "Hello I am Alice.",
                    "aliases": None,
                }
            ],
        }
        (cast_dir / "characters.json").write_text(json.dumps(cast_data))
        yield workdir


class TestAuditionCallback:
    def test_callback_triggers_retry_on_bad_output(self, audition_workdir):
        engine = MagicMock()
        engine.config = SimpleNamespace(seed=7)
        # alternate bad, good per call
        counter = {"n": 0}

        def design(text, instruct):
            counter["n"] += 1
            return (_bad() if counter["n"] % 2 == 1 else _clean()), SAMPLE_RATE

        engine.design_voice = MagicMock(side_effect=design)

        with patch("autiobook.callback.categorize_audio", _fake_categorize):
            with patch("autiobook.dramatize.create_tts_engine", return_value=engine):
                run_auditions(audition_workdir, callback=True)

        from autiobook.config import VOICE_EMOTIONS

        assert engine.design_voice.call_count == 2 * len(VOICE_EMOTIONS)
        assert engine.config.seed == 7 + len(VOICE_EMOTIONS)

    def test_without_callback_accepts_bad_output(self, audition_workdir):
        engine = MagicMock()
        engine.config = SimpleNamespace(seed=7)
        engine.design_voice = MagicMock(return_value=(_bad(), SAMPLE_RATE))

        with patch("autiobook.dramatize.create_tts_engine", return_value=engine):
            run_auditions(audition_workdir, callback=False)

        from autiobook.config import VOICE_EMOTIONS

        assert engine.design_voice.call_count == len(VOICE_EMOTIONS)
        assert engine.config.seed == 7

    def test_callback_raises_when_quality_never_passes(self, audition_workdir):
        engine = MagicMock()
        engine.config = SimpleNamespace(seed=1)
        engine.design_voice = MagicMock(return_value=(_bad(), SAMPLE_RATE))

        with patch("autiobook.callback.categorize_audio", _fake_categorize):
            with patch("autiobook.dramatize.create_tts_engine", return_value=engine):
                with pytest.raises(RuntimeError, match="failed audio quality"):
                    run_auditions(audition_workdir, callback=True)


class TestIntroduceCallback:
    def test_callback_triggers_retry_on_bad_output(self, audition_workdir):
        engine = MagicMock()
        engine.config = SimpleNamespace(seed=7)
        counter = {"n": 0}

        def design(text, instruct):
            counter["n"] += 1
            return (_bad() if counter["n"] % 2 == 1 else _clean()), SAMPLE_RATE

        engine.design_voice = MagicMock(side_effect=design)

        with patch("autiobook.callback.categorize_audio", _fake_categorize):
            with patch("autiobook.introduce.create_tts_engine", return_value=engine):
                run_introduce(audition_workdir, callback=True)

        # one character → two design calls (bad, then clean)
        assert engine.design_voice.call_count == 2
        assert engine.config.seed == 8

    def test_without_callback_accepts_bad_output(self, audition_workdir):
        engine = MagicMock()
        engine.config = SimpleNamespace(seed=7)
        engine.design_voice = MagicMock(return_value=(_bad(), SAMPLE_RATE))

        with patch("autiobook.introduce.create_tts_engine", return_value=engine):
            run_introduce(audition_workdir, callback=False)

        assert engine.design_voice.call_count == 1
        assert engine.config.seed == 7


def test_max_attempts_default():
    assert CALLBACK_MAX_ATTEMPTS == 5


class TestForensicArchival:
    def test_save_reject_writes_wav_and_sidecar(self):
        from autiobook.retake import save_reject

        with tempfile.TemporaryDirectory() as tmp:
            reject_dir = Path(tmp) / "rejected"
            wav = save_reject(
                reject_dir,
                _bad(),
                SAMPLE_RATE,
                ["silent"],
                "Alice/happy",
                metadata={"seed": 42, "attempt": 1},
            )
            assert wav.exists()
            sidecar = wav.with_suffix(".json")
            assert sidecar.exists()
            data = json.loads(sidecar.read_text())
            assert data["label"] == "Alice/happy"
            assert data["categories"] == ["silent"]
            assert data["sample_rate"] == SAMPLE_RATE
            assert data["metadata"]["seed"] == 42

    def test_generate_with_callback_archives_each_rejection(self):
        engine = SimpleNamespace(config=SimpleNamespace(seed=100))
        results = [
            (_bad(), SAMPLE_RATE),
            (_bad(), SAMPLE_RATE),
            (_clean(), SAMPLE_RATE),
        ]

        def gen():
            return results.pop(0)

        with tempfile.TemporaryDirectory() as tmp:
            reject_dir = Path(tmp) / "rejected"
            with patch("autiobook.callback.categorize_audio", _fake_categorize):
                generate_with_callback(
                    gen,
                    engine,
                    label="Alice/happy",
                    reject_dir=reject_dir,
                    metadata={"phase": "audition"},
                )
            wavs = list(reject_dir.glob(f"*{WAV_EXT}"))
            sidecars = list(reject_dir.glob("*.json"))
            assert len(wavs) == 2
            assert len(sidecars) == 2
            # verify metadata merge: phase + seed + attempt
            data = json.loads(sidecars[0].read_text())
            assert data["metadata"]["phase"] == "audition"
            assert "seed" in data["metadata"]
            assert "attempt" in data["metadata"]

    def test_audition_callback_archives_to_rejected_dir(self, audition_workdir):
        engine = MagicMock()
        engine.config = SimpleNamespace(seed=7)
        counter = {"n": 0}

        def design(text, instruct):
            counter["n"] += 1
            return (_bad() if counter["n"] % 2 == 1 else _clean()), SAMPLE_RATE

        engine.design_voice = MagicMock(side_effect=design)

        with patch("autiobook.callback.categorize_audio", _fake_categorize):
            with patch("autiobook.dramatize.create_tts_engine", return_value=engine):
                run_auditions(audition_workdir, callback=True)

        from autiobook.config import REJECTED_DIR, VOICE_EMOTIONS

        reject_dir = audition_workdir / "audition" / REJECTED_DIR
        wavs = list(reject_dir.glob(f"*{WAV_EXT}"))
        assert len(wavs) == len(VOICE_EMOTIONS)
        sidecars = list(reject_dir.glob("*.json"))
        data = json.loads(sidecars[0].read_text())
        assert data["metadata"]["phase"] == "audition"
        assert data["metadata"]["character"] == "Alice"
        assert "instruct" in data["metadata"]
