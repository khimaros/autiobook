from autiobook.dramatize import (
    _tokenize_with_positions,
    _validate_segments,
)
from autiobook.llm import ScriptSegment


class TestDetailedTokenization:
    """Detailed tests for _tokenize_with_positions logic."""

    def test_basic_tokenization(self):
        text = "Hello world."
        tokens = _tokenize_with_positions(text)
        assert len(tokens) == 2
        assert tokens[0] == ("hello", 0, 5)
        assert tokens[1] == ("world", 6, 11)

    def test_punctuation_handling(self):
        text = "Hello, world! How are you?"
        tokens = _tokenize_with_positions(text)
        words = [t[0] for t in tokens]
        assert words == ["hello", "world", "how", "are", "you"]

    def test_hyphenated_words(self):
        # Current implementation splits on non-word chars, so hyphen might split
        # Let's verify behavior. re.finditer(r"\w+") splits on hyphens.
        text = "near-religious experience"
        tokens = _tokenize_with_positions(text)
        words = [t[0] for t in tokens]
        # Expecting ["near", "religious", "experience"] based on \w+ regex
        assert words == ["near", "religious", "experience"]

    def test_contractions(self):
        # \w+ matches "don" and "t" separately if apostrophe is not \w
        # Python's \w usually includes alphanumeric + underscore.
        text = "don't won't"
        tokens = _tokenize_with_positions(text)
        words = [t[0] for t in tokens]
        # Expecting ["don", "t", "won", "t"] based on \w+
        assert words == ["don", "t", "won", "t"]

    def test_underscores(self):
        text = "snake_case_variable"
        tokens = _tokenize_with_positions(text)
        words = [t[0] for t in tokens]
        # Underscore is part of \w
        assert words == ["snake_case_variable"]

    def test_multiple_whitespace(self):
        text = "Word   Space\nNewline"
        tokens = _tokenize_with_positions(text)
        words = [t[0] for t in tokens]
        assert words == ["word", "space", "newline"]
        # Check positions for space handling
        assert tokens[1] == ("space", 7, 12)  # "Space" starts at index 7


class TestDetailedValidationLogic:
    """Detailed tests for _validate_segments missing/hallucination logic."""

    def test_perfect_match(self):
        source = "Hello world. This is a test."
        segments = [
            ScriptSegment("Narrator", "Hello world.", ""),
            ScriptSegment("Narrator", "This is a test.", ""),
        ]
        result = _validate_segments(source, segments)
        assert not result.missing
        assert not result.hallucinated

    def test_missing_start(self):
        source = "Intro text. Hello world."
        segments = [ScriptSegment("Narrator", "Hello world.", "")]
        result = _validate_segments(source, segments)
        assert len(result.missing) == 1
        assert result.missing[0][0] == "Intro text."
        assert result.missing[0][1] == 0  # insertion index

    def test_missing_end(self):
        source = "Hello world. Outro text."
        segments = [ScriptSegment("Narrator", "Hello world.", "")]
        result = _validate_segments(source, segments)
        assert len(result.missing) == 1
        assert result.missing[0][0] == "Outro text."
        assert result.missing[0][1] == 1  # insertion index (after last segment)

    def test_missing_middle(self):
        source = "Start. Middle. End."
        segments = [
            ScriptSegment("Narrator", "Start.", ""),
            ScriptSegment("Narrator", "End.", ""),
        ]
        result = _validate_segments(source, segments)
        assert len(result.missing) == 1
        assert result.missing[0][0] == "Middle."
        assert result.missing[0][1] == 1  # between seg 0 and 1

    def test_hallucination_detected(self):
        source = "Hello world."
        segments = [
            ScriptSegment("Narrator", "Hello world.", ""),
            ScriptSegment("Narrator", "I am a ghost.", ""),
        ]
        result = _validate_segments(source, segments)
        assert len(result.hallucinated) == 1
        assert result.hallucinated[0] == 1  # index of ghost segment

    def test_fuzzy_match_tolerance(self):
        # Minor differences should not trigger missing/hallucinated
        source = "Hello world!"
        # "Hello world." vs "Hello world!" -> tokens are "hello", "world" for both
        segments = [ScriptSegment("Narrator", "Hello world.", "")]
        result = _validate_segments(source, segments)
        assert not result.missing
        assert not result.hallucinated

    def test_significant_text_difference(self):
        source = "The quick brown fox."
        # "red" != "quick brown"
        segments = [ScriptSegment("Narrator", "The red fox.", "")]
        result = _validate_segments(source, segments)
        # Should flag "quick brown" as missing, and possibly segment as hallucinated if overlap low
        # "The", "fox" match (2/4 words). Ratio 0.5. Borderline.
        # Let's check logic: if ratio < 0.5 it's hallucinated. 0.5 is >= 0.5 so kept.
        # But "quick brown" should be missing.
        # And "red" is extra but we don't return "extra text", we return hallucinated *segments*.
        assert any("quick brown" in m[0] for m in result.missing)

    def test_completely_wrong_segment(self):
        source = "The quick brown fox."
        segments = [ScriptSegment("Narrator", "Jumps over dog.", "")]
        result = _validate_segments(source, segments)
        # 0 matches.
        assert len(result.hallucinated) == 1
        # entire source missing
        assert len(result.missing) >= 1

    def test_repeated_phrases_source(self):
        source = "Hello. Hello. Hello."
        segments = [
            ScriptSegment("Narrator", "Hello.", ""),
            ScriptSegment("Narrator", "Hello.", ""),
            ScriptSegment("Narrator", "Hello.", ""),
        ]
        result = _validate_segments(source, segments)
        assert not result.missing
        assert not result.hallucinated

    def test_missing_repetition(self):
        source = "Hello. Hello."
        segments = [ScriptSegment("Narrator", "Hello.", "")]
        result = _validate_segments(source, segments)
        # Should find one "Hello." missing
        assert len(result.missing) == 1

    def test_extra_repetition_segment(self):
        source = "Hello."
        segments = [
            ScriptSegment("Narrator", "Hello.", ""),
            ScriptSegment("Narrator", "Hello.", ""),
        ]
        result = _validate_segments(source, segments)
        # One segment matches "Hello", the other has 0 matches (consumed by first?)
        # Difflib usually matches greedily.
        # One segment will match, one will be hallucinated.
        assert len(result.hallucinated) == 1

    def test_empty_source(self):
        source = ""
        segments = [ScriptSegment("Narrator", "Something.", "")]
        result = _validate_segments(source, segments)
        assert len(result.hallucinated) == 1

    def test_empty_segments(self):
        source = "Text."
        segments = []
        result = _validate_segments(source, segments)
        assert len(result.missing) == 1
        assert result.missing[0][0] == "no segments provided"

    def test_mixed_content_boundary_cleanliness(self):
        # Ensure missing text doesn't include boundary punctuation from found segments
        source = "One. Two. Three."
        segments = [
            ScriptSegment("Narrator", "One.", ""),
            ScriptSegment("Narrator", "Three.", ""),
        ]
        result = _validate_segments(source, segments)
        assert len(result.missing) == 1
        # Should be "Two" or "Two." depending on punctuation handling
        # Logic trims punctuation from start/end of missing fragment
        assert "Two" in result.missing[0][0]
        # Should NOT contain "One." or "Three."

    def test_long_text_block(self):
        # Simulate a larger block to ensure difflib doesn't timeout/fail weirdly
        words = ["word"] * 100
        source = " ".join(words)
        segments = [ScriptSegment("Narrator", source, "")]
        result = _validate_segments(source, segments)
        assert not result.missing
        assert not result.hallucinated


class TestValidationOutputFormatting:
    """tests for format_validation_failure output."""

    def test_missing_fragment_shows_actual_text(self):
        from autiobook.dramatize import format_validation_failure

        source = "Start text. Missing middle. End text."
        segments = [
            ScriptSegment("Narrator", "Start text.", ""),
            ScriptSegment("Narrator", "End text.", ""),
        ]
        result = _validate_segments(source, segments)
        output = format_validation_failure(result, segments, source)
        # should show the actual missing fragment
        assert "Missing middle" in output

    def test_missing_fragment_shows_neighboring_segments(self):
        from autiobook.dramatize import format_validation_failure

        source = "First. Second. Third."
        segments = [
            ScriptSegment("Alice", "First.", ""),
            ScriptSegment("Bob", "Third.", ""),
        ]
        result = _validate_segments(source, segments)
        output = format_validation_failure(result, segments, source)
        # should show which segments are before/after
        assert "Alice" in output or "First" in output
        assert "Bob" in output or "Third" in output

    def test_missing_fragment_shows_full_line_context(self):
        from autiobook.dramatize import format_validation_failure

        source = "Context before. The missing part. Context after."
        segments = [
            ScriptSegment("Narrator", "Context before.", ""),
            ScriptSegment("Narrator", "Context after.", ""),
        ]
        result = _validate_segments(source, segments)
        output = format_validation_failure(result, segments, source)
        # should show full line for context
        assert "missing part" in output.lower()

    def test_hallucinated_segment_shows_text(self):
        from autiobook.dramatize import format_validation_failure

        source = "Only this exists."
        segments = [
            ScriptSegment("Narrator", "Only this exists.", ""),
            ScriptSegment("Ghost", "I am not real.", ""),
        ]
        result = _validate_segments(source, segments)
        output = format_validation_failure(result, segments, source)
        # should show the hallucinated segment's speaker and text
        assert "Ghost" in output
        assert "not real" in output

    def test_hallucinated_shows_segment_index(self):
        from autiobook.dramatize import format_validation_failure

        source = "Real text."
        segments = [
            ScriptSegment("Narrator", "Real text.", ""),
            ScriptSegment("Fake", "Hallucinated.", ""),
        ]
        result = _validate_segments(source, segments)
        output = format_validation_failure(result, segments, source)
        # should show segment index for easy identification
        assert "1" in output  # index of hallucinated segment

    def test_output_includes_insertion_position(self):
        from autiobook.dramatize import format_validation_failure

        source = "Before. Missing. After."
        segments = [
            ScriptSegment("Narrator", "Before.", ""),
            ScriptSegment("Narrator", "After.", ""),
        ]
        result = _validate_segments(source, segments)
        output = format_validation_failure(result, segments, source)
        # should indicate where the missing text should be inserted
        assert (
            "insert" in output.lower()
            or "between" in output.lower()
            or "after" in output.lower()
        )

    def test_empty_result_produces_empty_output(self):
        from autiobook.dramatize import format_validation_failure

        source = "All text present."
        segments = [ScriptSegment("Narrator", "All text present.", "")]
        result = _validate_segments(source, segments)
        output = format_validation_failure(result, segments, source)
        # no issues means empty or minimal output
        assert output == "" or "ok" in output.lower()


class TestRetainedSegments:
    """tests for Retained speaker handling (text kept but not narrated)."""

    def test_retained_segment_validates_as_covered(self):
        """retained segments should count toward source text coverage."""
        source = "[i] Chapter One. The story begins."
        segments = [
            ScriptSegment("Retained", "[i]", ""),
            ScriptSegment("Narrator", "Chapter One.", ""),
            ScriptSegment("Narrator", "The story begins.", ""),
        ]
        result = _validate_segments(source, segments)
        assert not result.missing
        assert not result.hallucinated

    def test_retained_roman_numeral(self):
        """roman numerals marked as Retained should pass validation."""
        source = "[ii] Second section."
        segments = [
            ScriptSegment("Retained", "[ii]", ""),
            ScriptSegment("Narrator", "Second section.", ""),
        ]
        result = _validate_segments(source, segments)
        assert not result.missing

    def test_retained_chapter_number(self):
        """chapter numbers marked as Retained should pass validation."""
        source = "2. The Next Chapter"
        segments = [
            ScriptSegment("Retained", "2.", ""),
            ScriptSegment("Narrator", "The Next Chapter", ""),
        ]
        result = _validate_segments(source, segments)
        assert not result.missing

    def test_unvoiced_backwards_compatibility(self):
        """'Unvoiced' should still work for backwards compatibility."""
        source = "[iii] Legacy content."
        segments = [
            ScriptSegment("Unvoiced", "[iii]", ""),
            ScriptSegment("Narrator", "Legacy content.", ""),
        ]
        result = _validate_segments(source, segments)
        assert not result.missing

    def test_missing_without_retained(self):
        """without Retained, section markers cause validation failures."""
        source = "[i] Chapter One."
        segments = [
            ScriptSegment("Narrator", "Chapter One.", ""),
        ]
        result = _validate_segments(source, segments)
        # [i] should be flagged as missing
        assert len(result.missing) == 1
        assert "[i]" in result.missing[0][0] or "i" in result.missing[0][0]
