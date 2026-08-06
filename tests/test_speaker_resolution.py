"""tests for --speaker voice reference resolution during synthesize.

--speaker names a cloned voice from audition/. when that reference wav is
absent the run must not quietly fall back to preset-voice synthesis: the
preset name is unrelated to what the user asked for, and --speaker has
already switched the engine to the clone model, which has no preset voices.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from autiobook.config import SAMPLE_RATE
from autiobook.tts import _perform_synthesis

CHAPTER_TEXT = "The first sentence. The second sentence. The third sentence."


class FakeEngine:
    """records which synthesis path the pipeline chose."""

    def __init__(self, voice=None, speaker="ryan"):
        self.config = SimpleNamespace(
            chunk_size=500,
            batch_size=4,
            seed=0,
            compile_model=False,
            voice=voice,
            speaker=speaker,
        )
        self.calls: list[tuple[str, str]] = []

    def synthesize(self, texts, instruct="", speaker=None):
        self.calls.append(("synthesize", speaker or self.config.speaker))
        return [np.zeros(SAMPLE_RATE, dtype=np.float32) for _ in texts], SAMPLE_RATE

    def clone_voice(self, texts, ref_audio, ref_text):
        self.calls.append(("clone", str(ref_audio)))
        return [np.zeros(SAMPLE_RATE, dtype=np.float32) for _ in texts], SAMPLE_RATE


@pytest.fixture
def workdir(tmp_path):
    """a workdir with one extracted chapter ready to synthesize."""
    extract = tmp_path / "extract"
    extract.mkdir()
    synth = tmp_path / "synthesize"
    synth.mkdir()
    txt = extract / "01_One.txt"
    txt.write_text(CHAPTER_TEXT, encoding="utf-8")
    return tmp_path, [(txt, synth / "01_One.wav")]


def add_audition(workdir_path: Path, name: str) -> Path:
    """write an audition reference wav for `name`."""
    audition = workdir_path / "audition"
    audition.mkdir(exist_ok=True)
    wav = audition / f"{name}.wav"
    sf.write(str(wav), np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    return wav


class TestSpeakerResolution:
    def test_clones_when_audition_wav_exists(self, workdir):
        """the working case: --speaker resolves to its audition reference."""
        path, pending = workdir
        add_audition(path, "Narrator")
        engine = FakeEngine(voice="Narrator")

        _perform_synthesis(engine, pending)

        assert engine.calls, "nothing was synthesized"
        assert all(
            kind == "clone" for kind, _ in engine.calls
        ), f"expected cloning, got {engine.calls}"

    def test_missing_audition_wav_is_an_error(self, workdir):
        """the reported bug: a missing reference must not degrade silently.

        today this falls through to engine.synthesize() with the unrelated
        --voice default, which the clone model rejects with an opaque
        'unknown voice' http 400 several thousand segments into the run.
        """
        _, pending = workdir
        engine = FakeEngine(voice="Narrator")

        with pytest.raises(FileNotFoundError, match="Narrator"):
            _perform_synthesis(engine, pending)

    def test_no_speaker_uses_preset_voice(self, workdir):
        """without --speaker, preset synthesis is still the correct path."""
        _, pending = workdir
        engine = FakeEngine(voice=None, speaker="ryan")

        _perform_synthesis(engine, pending)

        assert engine.calls
        assert all(kind == "synthesize" for kind, _ in engine.calls)
