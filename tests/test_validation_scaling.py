import time

from autiobook.dramatize import _validate_segments
from autiobook.llm import ScriptSegment


class TestValidationScaling:
    """Tests for validation logic on larger text inputs (simulating full chapters)."""

    def generate_large_input(self, word_count=5000):
        """Generates deterministic large text and corresponding segments."""
        words = []
        segments = []

        # Pattern: 10 words narration, 5 words dialogue
        for i in range(0, word_count, 15):
            # Narration
            narrative = (
                f"Narration sequence number {i} keeping the story moving forward."
            )
            words.extend(narrative.split())
            segments.append(ScriptSegment("Narrator", narrative, ""))

            # Dialogue
            dialogue = f"Dialogue {i} says hello."
            words.extend(dialogue.split())
            segments.append(ScriptSegment("Character", dialogue, ""))

        source_text = " ".join(words)
        return source_text, segments

    def test_large_chapter_validation_performance(self):
        """Verify validation of a 5000-word chapter finishes quickly (< 1s)."""
        source_text, segments = self.generate_large_input(word_count=5000)

        start_time = time.time()
        result = _validate_segments(source_text, segments)
        duration = time.time() - start_time

        assert not result.missing
        assert not result.hallucinated
        # 5000 words should be handled efficiently by difflib
        print(f"\nValidation of 5000 words took {duration:.4f}s")
        assert duration < 2.0, f"Validation took too long: {duration:.4f}s"

    def test_large_chapter_with_missing_text(self):
        """Verify missing text detection works correctly in a large document."""
        source_text, segments = self.generate_large_input(word_count=5000)

        # Introduce a missing segment in the middle (approx index 330)
        missing_idx = len(segments) // 2
        missing_seg = segments.pop(missing_idx)
        missing_text = missing_seg.text

        # The source text still has it, but segments don't.
        # Note: generate_large_input constructs source from segments list,
        # so if we pop from segments AFTER generating source, source has it, segments don't.
        # This is the correct setup for "missing text".

        result = _validate_segments(source_text, segments)

        assert len(result.missing) == 1
        # Fuzzy match might return slightly different boundaries depending on tokenization,
        # but the core text should be found.
        # The missing text is "Narration sequence number ... " or "Dialogue ..."
        assert missing_text.strip(" .") in result.missing[0][0]
        assert result.missing[0][1] == missing_idx  # insertion index

    def test_large_chapter_with_hallucination(self):
        """Verify hallucination detection works in large document."""
        source_text, segments = self.generate_large_input(word_count=5000)

        # Insert a hallucinated segment
        hallucination = ScriptSegment(
            "Narrator", "This text does not exist in the source anywhere.", ""
        )
        insert_idx = len(segments) // 2
        segments.insert(insert_idx, hallucination)

        result = _validate_segments(source_text, segments)

        assert len(result.hallucinated) == 1
        assert result.hallucinated[0] == insert_idx

    def test_very_large_chapter_limit(self):
        """Stress test with 20,000 words (approx novella size)."""
        # This ensures O(N^2) behavior of difflib (if any) doesn't explode
        source_text, segments = self.generate_large_input(word_count=20000)

        start_time = time.time()
        result = _validate_segments(source_text, segments)
        duration = time.time() - start_time

        print(f"\nValidation of 20,000 words took {duration:.4f}s")
        # Allow more time, but should still be reasonable
        assert duration < 10.0
        assert not result.missing
        assert not result.hallucinated
