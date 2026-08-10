"""tests for the interactive directed-audition loop.

the loop is hard to drive end to end, but its bookkeeping around the
per-character skip path is not, and that is where it has crashed.
"""

from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from autiobook.audition import _run_directed_design, run_audition
from autiobook.config import AUDITION_SAMPLE_LINE, SAMPLE_RATE
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
        text = audition_line or AUDITION_SAMPLE_LINE
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
                voice="Male, sixties, low rasping baritone.",
            ),
            Character(
                name="Molly",
                description="A street samurai.",
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


CAST = [
    Character(
        name="Ratz",
        description="A gruff bartender with a prosthetic arm.",
        aliases=["the bartender"],
        voice="Male, sixties, low rasping baritone.",
    ),
    Character(
        name="Molly",
        description="A street samurai with mirrored lenses.",
        voice="Female, thirties, flat and clipped.",
    ),
]


class TestSeedResetBetweenCharacters:
    """each character starts from the configured seed, not the last random one."""

    def test_a_rolled_seed_does_not_leak_into_the_next_character(self, tmp_path):
        """[n]ext rolls config.seed mid-character; the next one must not inherit it."""
        from autiobook.audition import _run_directed_design

        BASE, ROLLED = 4242, 123456789
        engine = FakeEngine()
        engine.config.seed = BASE
        # every character is cached, so the loop runs its per-character setup
        # and skips -- enough to observe what the seed is set to each time.
        voices = seed_workdir(tmp_path, CAST)

        seen: list[int] = []

        class SpyConfig:
            """records each seed assignment, and rolls one as [n]ext would."""

            def __init__(self, inner):
                object.__setattr__(self, "_inner", inner)

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __setattr__(self, name, value):
                setattr(self._inner, name, value)
                if name != "seed":
                    return
                seen.append(value)
                if len(seen) == 1:
                    object.__setattr__(self._inner, "seed", ROLLED)

        _run_directed_design(tmp_path, CAST, engine, SpyConfig(engine.config), voices)

        assert seen == [BASE, BASE], "second character inherited the rolled seed"

    @pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
    def test_first_take_uses_the_configured_seed(self, tmp_path, streaming):
        """both paths open a character on the configured seed, not a random one.

        the buffered path rolled its own seed for every take, so which one a
        character opened on depended on whether streaming was available."""
        from autiobook.audition import recorded_seed

        BASE = 4242
        engine = FakeEngine()
        engine.config.seed = BASE
        if streaming:
            engine.streaming = True
            # only its presence matters: with no pcm sink the synth falls
            # through to design_voice
            engine.design_voice_stream = lambda **kw: None

        voices = tmp_path / "audition"
        voices.mkdir(parents=True)

        with patch("autiobook.casting._play_pcm_stream", return_value=None):
            with patch("autiobook.casting._play_wav_async", return_value=None):
                with patch("autiobook.casting._stop_playback"):
                    with patch("autiobook.audition.prompt_choice", return_value="y"):
                        _run_directed_design(
                            tmp_path, CAST, engine, engine.config, voices
                        )

        for char in CAST:
            assert recorded_seed(tmp_path, char.name) == BASE

    def test_base_seed_is_read_before_the_loop_mutates_it(self, tmp_path):
        """the reset value is the seed as configured, not whatever ran last."""
        from autiobook.audition import _run_directed_design

        engine = FakeEngine()
        engine.config.seed = 4242
        voices = seed_workdir(tmp_path, CAST)

        _run_directed_design(tmp_path, CAST, engine, engine.config, voices)

        assert engine.config.seed == 4242


class TestCastConfirmation:
    """--directed shows the roster and waits for approval before any takes."""

    def _run(self, tmp_path, answer, **kwargs):
        with patch("autiobook.audition.prompt_choice", return_value=answer) as ask:
            with patch("autiobook.audition._run_directed_design") as design:
                with patch("autiobook.audition._run_preset") as preset:
                    with patch("autiobook.audition.create_tts_engine"):
                        run_audition(tmp_path, cast=CAST, directed=True, **kwargs)
        return ask, design, preset

    def test_roster_is_shown_in_full(self, tmp_path, capsys):
        self._run(tmp_path, "y")

        out = capsys.readouterr().out
        assert "cast: 2 characters" in out
        for char in CAST:
            assert char.name in out
            assert char.description in out
        assert "the bartender" in out  # aliases surface too

    def test_yes_proceeds(self, tmp_path):
        _, design, _ = self._run(tmp_path, "y")
        assert design.call_count == 1

    def test_enter_proceeds(self, tmp_path):
        _, design, _ = self._run(tmp_path, "")
        assert design.call_count == 1

    def test_quit_aborts_before_any_take(self, tmp_path, capsys):
        _, design, preset = self._run(tmp_path, "q")

        assert design.call_count == 0
        assert preset.call_count == 0
        assert "cancelled" in capsys.readouterr().out

    def test_preset_casting_is_gated_too(self, tmp_path):
        _, _, preset = self._run(tmp_path, "q", preset_voices=True)
        assert preset.call_count == 0

    def test_approved_voices_are_left_off_the_roster(self, tmp_path, capsys):
        """already-approved characters are not work, so they are not shown."""
        seed_workdir(tmp_path, [CAST[0]])

        self._run(tmp_path, "y")

        out = capsys.readouterr().out
        assert "1 characters to audition (1 already approved)" in out
        assert "Molly" in out
        assert "Ratz" not in out

    def test_nothing_pending_does_not_prompt(self, tmp_path):
        seed_workdir(tmp_path, CAST)

        ask, design, _ = self._run(tmp_path, "q")

        assert ask.call_count == 0
        assert design.call_count == 1

    def test_same_cast_is_not_reprompted(self, tmp_path):
        """approval sticks to the cast, so a resumed session goes back to work."""
        ask, _, _ = self._run(tmp_path, "y")
        assert ask.call_count == 1

        ask, design, _ = self._run(tmp_path, "q")

        assert ask.call_count == 0, "re-prompted for an unchanged cast"
        assert design.call_count == 1

    def test_edited_cast_is_reprompted(self, tmp_path):
        from dataclasses import replace

        self._run(tmp_path, "y")

        edited = [replace(CAST[0], description="Now a retired bartender."), CAST[1]]
        with patch("autiobook.audition.prompt_choice", return_value="y") as ask:
            with patch("autiobook.audition._run_directed_design"):
                with patch("autiobook.audition.create_tts_engine"):
                    run_audition(tmp_path, cast=edited, directed=True)

        assert ask.call_count == 1

    def test_preset_skips_characters_already_cast(self, tmp_path, capsys):
        from autiobook.casting import save_voices

        save_voices(tmp_path, {"Ratz": "ash"})

        self._run(tmp_path, "y", preset_voices=True)

        out = capsys.readouterr().out
        assert "1 characters to audition (1 already approved)" in out
        assert "Molly" in out

    def test_undirected_run_is_not_prompted(self, tmp_path):
        with patch("autiobook.audition.prompt_choice") as ask:
            with patch("autiobook.audition._run_preset"):
                run_audition(tmp_path, cast=CAST, preset_voices=True)

        assert ask.call_count == 0
