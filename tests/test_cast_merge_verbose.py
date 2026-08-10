"""tests for merging llm cast output into an existing cast.

the merge decides what actually lands in characters.json, and --verbose is how
that gets reviewed, so both the mutation and the reporting are pinned here.
"""

from autiobook.dramatize import _merge_character_into_cast
from autiobook.llm import Character


def build(**kw):
    base = {
        "name": "Ratz",
        "description": "A gruff bartender.",
        "voice": "Male, sixties, low rasping baritone.",
    }
    base.update(kw)
    return Character(**base)


def make_cast(existing):
    cast_map = {existing.name.lower(): existing}
    alias_map = {a.lower(): existing.name.lower() for a in (existing.aliases or [])}
    return cast_map, alias_map


class TestVoiceMerge:
    def test_updated_voice_lands(self):
        """the reported gap: a refined voice prompt was dropped on the floor."""
        existing = build()
        cast_map, alias_map = make_cast(existing)
        incoming = build(voice="Male, sixties, wet rasping baritone, slow.")

        result = _merge_character_into_cast(incoming, cast_map, alias_map)

        assert existing.voice == "Male, sixties, wet rasping baritone, slow."
        assert result == "updated"

    def test_voice_update_is_reported(self, capsys):
        existing = build()
        cast_map, alias_map = make_cast(existing)
        _merge_character_into_cast(
            build(voice="Male, sixties, wet rasping baritone."),
            cast_map,
            alias_map,
            verbose=True,
        )
        out = capsys.readouterr().out
        assert "voice:" in out
        assert "low rasping baritone" in out  # the old value
        assert "wet rasping baritone" in out  # the new value

    def test_unchanged_voice_is_not_a_diff(self):
        existing = build()
        cast_map, alias_map = make_cast(existing)
        assert _merge_character_into_cast(build(), cast_map, alias_map) == "unchanged"

    def test_pre_split_existing_compares_against_its_description(self):
        """an existing character with no voice uses description as the prompt."""
        existing = Character(name="Ratz", description="Low rasping baritone.")
        cast_map, alias_map = make_cast(existing)
        # the line is held equal so only the voice/description path is in play
        incoming = build(
            description="Low rasping baritone.",
            voice="Low rasping baritone.",
        )

        result = _merge_character_into_cast(incoming, cast_map, alias_map)

        assert result == "unchanged"
        assert existing.voice_prompt() == "Low rasping baritone."


class TestMatchLabelling:
    def test_reemitting_own_aliases_is_an_update_not_a_merge(self, capsys):
        """matching on an alias the character already owns is not a merge."""
        existing = build(aliases=["the bartender"])
        cast_map, alias_map = make_cast(existing)
        result = _merge_character_into_cast(
            build(aliases=["the bartender"], voice="Male, sixties, guttural."),
            cast_map,
            alias_map,
            verbose=True,
        )
        assert result == "updated"
        assert "updated 'Ratz'" in capsys.readouterr().out

    def test_a_different_name_matching_an_alias_is_a_merge(self):
        existing = build(aliases=["the bartender"])
        cast_map, alias_map = make_cast(existing)
        incoming = build(name="The Bartender", aliases=None)
        assert _merge_character_into_cast(incoming, cast_map, alias_map) == "merged"


class TestReporting:
    def test_new_character_shows_its_voice(self, capsys):
        cast_map: dict = {}
        alias_map: dict = {}
        _merge_character_into_cast(build(), cast_map, alias_map, verbose=True)
        out = capsys.readouterr().out
        assert "added new character" in out
        assert "voice: 'Male, sixties, low rasping baritone.'" in out
        assert "description: 'A gruff bartender.'" in out

    def test_unchanged_character_is_named(self, capsys):
        existing = build()
        cast_map, alias_map = make_cast(existing)
        _merge_character_into_cast(build(), cast_map, alias_map, verbose=True)
        assert "unchanged 'Ratz'" in capsys.readouterr().out

    def test_quiet_by_default(self, capsys):
        existing = build()
        cast_map, alias_map = make_cast(existing)
        _merge_character_into_cast(build(voice="New voice."), cast_map, alias_map)
        assert capsys.readouterr().out == ""


class TestAliasStopwords:
    """a pronoun alias captures every segment the script attributes to it."""

    def test_pronoun_is_dropped_from_a_new_character(self):
        cast_map: dict = {}
        alias_map: dict = {}

        _merge_character_into_cast(
            build(aliases=["the bartender", "he", "him"]), cast_map, alias_map
        )

        assert cast_map["ratz"].aliases == ["the bartender"]
        assert "he" not in alias_map

    def test_pronoun_is_dropped_from_an_update(self):
        existing = build()
        cast_map, alias_map = make_cast(existing)

        _merge_character_into_cast(build(aliases=["she"]), cast_map, alias_map)

        assert existing.aliases is None
        assert "she" not in alias_map

    def test_stored_pronoun_is_cleaned_on_the_next_pass(self):
        """casts already carrying one are fixed rather than grandfathered."""
        existing = build(aliases=["the bartender", "her"])
        cast_map, alias_map = make_cast(existing)

        _merge_character_into_cast(build(), cast_map, alias_map)

        assert existing.aliases == ["the bartender"]

    def test_multiword_alias_containing_a_pronoun_survives(self):
        """only whole aliases match: 'her ladyship' is a naming form."""
        cast_map: dict = {}
        alias_map: dict = {}

        _merge_character_into_cast(
            build(aliases=["her ladyship", "the old man"]), cast_map, alias_map
        )

        assert cast_map["ratz"].aliases == ["her ladyship", "the old man"]

    def test_punctuated_and_capitalised_pronouns_are_caught(self):
        cast_map: dict = {}
        alias_map: dict = {}

        _merge_character_into_cast(
            build(aliases=["He.", " THEY "]), cast_map, alias_map
        )

        assert cast_map["ratz"].aliases is None

    def test_prompt_forbids_pronouns(self, cast_prompt):
        prompt, _ = cast_prompt()

        assert "NEVER a pronoun" in prompt
        for word in ('"he"', '"she"', '"them"'):
            assert word in prompt
