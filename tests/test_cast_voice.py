"""tests for the character/voice description split.

the design prompt sent to VoiceDesign is kept separate from who the character
is, so it stays purely acoustic. casts written before the split carry a
combined blob in `description` and must keep generating the same voices.
"""

import json

from autiobook.config import CAST_FILE, DEFAULT_CAST
from autiobook.dramatize import load_cast, save_cast
from autiobook.llm import Character, _parse_cast_list, _validate_cast_list


def write_cast(workdir, characters, version=5):
    d = workdir / "cast"
    d.mkdir(parents=True, exist_ok=True)
    (d / CAST_FILE).write_text(
        json.dumps({"version": version, "characters": characters}), encoding="utf-8"
    )


class TestVoicePrompt:
    def test_voice_is_used_when_present(self):
        c = Character(
            name="X",
            description="A tired medic.",
            voice="Low, gravelly.",
        )
        assert c.voice_prompt() == "Low, gravelly."

    def test_falls_back_to_description(self):
        """a pre-split cast has its design prompt in description."""
        c = Character(name="X", description="Low, gravelly voice.")
        assert c.voice_prompt() == "Low, gravelly voice."


class TestCastRoundTrip:
    def test_voice_survives_save_and_load(self, tmp_path):
        cast = [
            Character(
                name="Mira",
                description="A burnt-out field medic.",
                voice="Female, late twenties, low pitch.",
            )
        ]
        save_cast(tmp_path, cast)
        loaded = load_cast(tmp_path)
        assert loaded[0].description == "A burnt-out field medic."
        assert loaded[0].voice == "Female, late twenties, low pitch."

    def test_pre_split_cast_keeps_its_design_prompt(self, tmp_path):
        """the migration that matters: no voice key means description is it."""
        write_cast(
            tmp_path,
            [
                {
                    "name": "Ratz",
                    "description": "Gruff bartender; low, wet, rasping baritone.",
                }
            ],
            version=4,
        )
        c = load_cast(tmp_path)[0]
        assert c.voice == ""
        assert c.voice_prompt() == "Gruff bartender; low, wet, rasping baritone."

    def test_saving_a_migrated_cast_records_the_prompt(self, tmp_path):
        """re-saving must not blank the design prompt of a pre-split cast."""
        write_cast(
            tmp_path,
            [
                {
                    "name": "Ratz",
                    "description": "Low rasping baritone.",
                }
            ],
            version=4,
        )
        save_cast(tmp_path, load_cast(tmp_path))
        stored = json.loads((tmp_path / "cast" / CAST_FILE).read_text())
        assert stored["characters"][0]["voice"] == "Low rasping baritone."

    def test_legacy_list_format_still_loads(self, tmp_path):
        d = tmp_path / "cast"
        d.mkdir(parents=True)
        (d / CAST_FILE).write_text(
            json.dumps([{"name": "A", "description": "Soft."}]),
            encoding="utf-8",
        )
        assert load_cast(tmp_path)[0].voice_prompt() == "Soft."

    def test_default_cast_carries_both_fields(self, tmp_path):
        cast = load_cast(tmp_path)  # no file -> DEFAULT_CAST
        assert len(cast) == len(DEFAULT_CAST)
        for c in cast:
            assert c.description and c.voice
            assert c.voice != c.description


class TestParsing:
    def test_voice_is_parsed(self):
        chars = _parse_cast_list(
            {
                "characters": [
                    {
                        "name": "Mira",
                        "description": "A medic.",
                        "voice": "Female, low pitch.",
                    }
                ]
            }
        )
        assert chars[0].voice == "Female, low pitch."

    def test_short_key_is_accepted(self):
        chars = _parse_cast_list(
            {"characters": [{"n": "M", "d": "A medic.", "v": "Low.", "a": "Hi."}]}
        )
        assert chars[0].voice == "Low."

    def test_missing_voice_is_a_validation_error(self):
        errors = _validate_cast_list([Character(name="M", description="A medic.")])
        assert any("voice" in e for e in errors)

    def test_complete_character_validates(self):
        errors = _validate_cast_list(
            [
                Character(
                    name="M",
                    description="A medic.",
                    voice="Low, clipped.",
                )
            ]
        )
        assert errors == []


class TestVoicePromptDimensions:
    """the voice spec tracks what Qwen3-TTS VoiceDesign documents reading.

    measured across 75 audition takes, volume descriptors moved rms by 1.3 dB
    between "quiet" and "loud" (se 1.17) -- indistinguishable from noise, and
    volume is absent from the vendor's list of controllable dimensions.
    """

    def test_documented_dimensions_are_named(self, cast_prompt):
        prompt, _ = cast_prompt()

        for dimension in ("gender:", "age:", "pitch:", "speed:", "timbre:", "emotion:"):
            assert dimension in prompt

    def test_documented_vocabulary_is_offered(self, cast_prompt):
        """the model's own words, as a starting point rather than a ceiling."""
        prompt, _ = cast_prompt()

        for word in ("slightly high", "slightly fast", "husky", "resonant", "composed"):
            assert word in prompt

    def test_accent_is_required_not_conditional(self, cast_prompt):
        """making it conditional emptied it: 0/8 of a translated novel's cast
        got an accent, because no dialect is rendered in the prose."""
        prompt, _ = cast_prompt()

        assert "accent: always name one" in prompt

    def test_distinctiveness_is_demanded(self, cast_prompt):
        """closed vocabularies collide: two middle-aged men landed on the same
        five words, differing only in speed."""
        prompt, _ = cast_prompt()

        assert "audibly nobody else" in prompt
        assert "add a short distinguishing phrase" in prompt

    def test_volume_is_ruled_out(self, cast_prompt):
        prompt, _ = cast_prompt()

        assert "Do NOT describe volume or loudness" in prompt

    def test_example_voice_uses_the_vocabulary(self, cast_prompt):
        """the example is the strongest steer, so it must not model the old style."""
        prompt, _ = cast_prompt()
        start = prompt.index('"voice": "Female')
        example = prompt[start : prompt.index('"aliases"', start)]

        assert "volume" not in example.lower()
        for word in ("young adult", "slightly low pitch", "husky timbre"):
            assert word in example
