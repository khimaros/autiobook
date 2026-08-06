"""tests for the interactive directed-audition loop.

the loop is hard to drive end to end, but its bookkeeping around the
per-character skip path is not, and that is where it has crashed.
"""

import numpy as np
import soundfile as sf

from autiobook.audition import _run_directed_design
from autiobook.config import SAMPLE_RATE
from autiobook.llm import Character
from autiobook.resume import ResumeManager, compute_hash


class FakeEngine:
    """an engine that must never be asked to synthesize."""

    def __init__(self):
        self.config = type("cfg", (), {"stream_batch_size": 0, "seed": 1})()
        self.calls = 0

    def design_voice(self, text, instruct):
        self.calls += 1
        return np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE


def seed_workdir(tmp_path, cast, audition_line=None):
    """mark every character already auditioned, as a resumed run would be."""
    voices = tmp_path / "audition"
    voices.mkdir(parents=True, exist_ok=True)
    resume = ResumeManager.for_command(tmp_path, "audition")
    for char in cast:
        sf.write(
            str(voices / f"{char.name}.wav"),
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            SAMPLE_RATE,
        )
        text = audition_line or char.audition_line
        resume.update(
            char.name,
            compute_hash(
                {"name": char.name, "description": char.voice_prompt(), "text": text}
            ),
        )
    resume.save()
    return voices


class TestAllCharactersFresh:
    def test_does_not_crash_when_everything_is_skipped(self, tmp_path, capsys):
        """resuming a finished audition skips every character.

        the quit flag was initialised inside the per-character loop, after the
        skip `continue`, so a fully-cached run reached the post-loop check with
        the name unbound and died with UnboundLocalError.
        """
        cast = [
            Character(
                name="Ratz",
                description="A gruff bartender.",
                audition_line="Sure, kid.",
                voice="Male, sixties, low rasping baritone.",
            ),
            Character(
                name="Molly",
                description="A street samurai.",
                audition_line="Don't.",
                voice="Female, thirties, flat and clipped.",
            ),
        ]
        voices = seed_workdir(tmp_path, cast)
        engine = FakeEngine()

        _run_directed_design(tmp_path, cast, engine, engine.config, voices)

        out = capsys.readouterr().out
        assert "0 accepted, 2 skipped" in out
        assert engine.calls == 0, "a cached character was re-synthesized"

    def test_empty_cast_does_not_crash(self, tmp_path, capsys):
        voices = tmp_path / "audition"
        voices.mkdir(parents=True)
        engine = FakeEngine()

        _run_directed_design(tmp_path, [], engine, engine.config, voices)

        assert "0 accepted, 0 skipped" in capsys.readouterr().out
