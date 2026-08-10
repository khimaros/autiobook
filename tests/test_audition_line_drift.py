"""tests for the fixed audition line.

by default every reference clip speaks the same sentence, so clips differ only
in voice. the llm is told not to propose one and any it proposes is dropped on
parse unless --llm-audition-lines is set, a hand-written line always wins, and
perform reads what a clip actually said from the record written when it was
rendered.
"""

import numpy as np
import soundfile as sf  # type: ignore

from autiobook.audition import (
    AUDITION_COMMAND,
    audition_task_hash,
    recorded_audition_lines,
)
from autiobook.config import AUDITION_SAMPLE_LINE, SAMPLE_RATE, VOICE_EMOTIONS
from autiobook.dramatize import character_hash
from autiobook.llm import Character
from autiobook.resume import ResumeManager


def _character(**kwargs):
    base = dict(
        name="Obligator",
        description="A senior Ministry obligator.",
        voice="Male, middle-aged, low pitch, detached tone.",
    )
    base.update(kwargs)
    return Character(**base)


class TestTheLineIsUserSuppliedOnly:
    """the llm never sets it; a person editing characters.json does."""

    def test_default_is_the_shared_sample_line(self):
        assert _character().audition_text() == AUDITION_SAMPLE_LINE

    def test_sample_line_is_the_neutral_emotion_line(self):
        """the base clip is the neutral identity; emote clips already used
        their own emotion's line, so this finishes that pattern."""
        assert AUDITION_SAMPLE_LINE == VOICE_EMOTIONS["neutral"][1]

    def test_hand_edited_line_is_honored(self):
        mine = "Only I would say a thing like that."

        assert _character(audition_line=mine).audition_text() == mine

    def test_run_wide_override_wins(self):
        """--audition-line is documented as applying to all characters."""
        mine = "Only I would say a thing like that."
        override = "Everyone says this one."

        assert _character(audition_line=mine).audition_text(override) == override

    def test_llm_proposal_is_dropped_on_parse(self):
        from autiobook.llm import _parse_cast_list

        chars = _parse_cast_list(
            [
                {
                    "name": "Obligator",
                    "description": "d",
                    "voice": "v",
                    "audition_line": "Something the model invented.",
                }
            ]
        )

        assert len(chars) == 1
        assert chars[0].audition_line == ""
        assert chars[0].audition_text() == AUDITION_SAMPLE_LINE

    def test_a_missing_line_is_not_a_validation_error(self):
        from autiobook.llm import _validate_cast_list

        assert _validate_cast_list([_character()]) == []


class TestLlmAuditionLinesFlag:
    """--llm-audition-lines asks for a line; off, the llm is told not to."""

    def test_off_by_default_the_prompt_forbids_it(self, cast_prompt):
        prompt, chars = cast_prompt()

        assert "do NOT invent an audition_line" in prompt
        assert chars[0].audition_line == ""

    def test_on_the_prompt_asks_for_it_and_it_is_parsed(self, cast_prompt):
        prompt, chars = cast_prompt(audition_lines=True)

        assert "do NOT invent an audition_line" not in prompt
        assert "- audition_line:" in prompt
        assert chars[0].audition_line == "Model wrote this."


class TestHandEditedLineSurvives:
    """a line set by hand has to outlive a cast re-run and a merge."""

    def test_round_trips_through_the_cast_file(self, tmp_path):
        from autiobook.dramatize import load_cast, save_cast

        mine = "Only I would say a thing like that."
        save_cast(tmp_path, [_character(audition_line=mine)])

        assert load_cast(tmp_path)[0].audition_line == mine

    def test_llm_reemission_does_not_clear_it(self):
        from autiobook.dramatize import _merge_character_into_cast

        mine = "Only I would say a thing like that."
        existing = _character(audition_line=mine)
        cast_map = {existing.name.lower(): existing}

        # what the llm emits now: no line at all
        _merge_character_into_cast(
            _character(description="Refined."), cast_map, {}, verbose=False
        )

        assert cast_map["obligator"].audition_line == mine

    def test_design_text_overwrites_deliberately(self):
        """`design --text` is a person changing their mind; it may overwrite."""
        from autiobook.dramatize import _merge_character_into_cast

        existing = _character(audition_line="Old line.")
        cast_map = {existing.name.lower(): existing}

        result = _merge_character_into_cast(
            _character(audition_line="New line."),
            cast_map,
            {},
            verbose=False,
            overwrite_audition_line=True,
        )

        assert cast_map["obligator"].audition_line == "New line."
        assert result == "updated"

    def test_llm_line_fills_an_empty_slot(self):
        from autiobook.dramatize import _merge_character_into_cast

        existing = _character()
        cast_map = {existing.name.lower(): existing}

        _merge_character_into_cast(
            _character(audition_line="Model wrote this."), cast_map, {}, verbose=False
        )

        assert cast_map["obligator"].audition_line == "Model wrote this."

    def test_llm_line_never_overwrites_a_hand_written_one(self):
        from autiobook.dramatize import _merge_character_into_cast

        mine = "Only I would say a thing like that."
        existing = _character(audition_line=mine)
        cast_map = {existing.name.lower(): existing}

        result = _merge_character_into_cast(
            _character(audition_line="Model wrote this."), cast_map, {}, verbose=False
        )

        assert cast_map["obligator"].audition_line == mine
        assert result == "unchanged"


class TestPerformCacheKeyedOnVoiceOnly:
    def test_char_hash_tracks_the_voice(self):
        assert character_hash(_character()) != character_hash(
            _character(voice="Female, high.")
        )

    def test_char_hash_is_stable_across_description_only_edits(self):
        """description is not sent to the tts model; voice_prompt is."""
        assert character_hash(_character()) == character_hash(
            _character(description="Reworded, same voice.")
        )


class TestRefTextFollowsTheAudio:
    """--audition-line renders clips that say something else; ref_text has to
    describe the audio, and the record is what knows."""

    def _workdir(self, tmp_path, recorded):
        audition = tmp_path / AUDITION_COMMAND
        audition.mkdir(parents=True)
        sf.write(
            str(audition / "Obligator.wav"),
            np.zeros(SAMPLE_RATE // 10, dtype="float32"),
            SAMPLE_RATE,
        )
        resume = ResumeManager.for_command(tmp_path, AUDITION_COMMAND)
        resume.update(
            "Obligator",
            audition_task_hash("Obligator", "voice", recorded),
            character="Obligator",
            audition_line=recorded,
        )
        resume.save()
        return tmp_path

    def test_recorded_line_is_returned(self, tmp_path):
        override = "A line supplied with --audition-line."
        self._workdir(tmp_path, override)

        assert recorded_audition_lines(tmp_path)["Obligator"] == override

    def test_missing_record_is_absent(self, tmp_path):
        (tmp_path / AUDITION_COMMAND).mkdir(parents=True)

        assert recorded_audition_lines(tmp_path) == {}

    def test_cast_approval_entry_is_not_a_character(self, tmp_path):
        """the roster-approval record shares the namespace but has no line."""
        from autiobook.audition import CAST_APPROVAL_KEY

        self._workdir(tmp_path, AUDITION_SAMPLE_LINE)
        resume = ResumeManager.for_command(tmp_path, AUDITION_COMMAND)
        resume.update(CAST_APPROVAL_KEY, "roster-hash")
        resume.save()

        assert CAST_APPROVAL_KEY not in recorded_audition_lines(tmp_path)
