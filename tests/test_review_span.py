"""tests for the source span review builds for each batch.

_locate_span decides which slice of the source the review llm sees and which
slice the corrected batch is validated against. if the span omits text a
segment covers, that segment can only ever come back "hallucinated", so the
batch is rejected no matter what the llm returns.
"""

from autiobook.dramatize import _locate_span

# the source uses a typographic apostrophe, as real epubs do
SOURCE = (
    "This book is a work of fiction. Names, characters, places, and incidents "
    "are the product of the author’s imagination or are used fictitiously.\n\n"
    "Copyright 2024 by Daniel Abraham and Ty Franck\n\n"
    "Cover design by Lauren Panepinto\n\n"
    "Hachette Book Group supports the right to free expression."
)

# the script carries a straight apostrophe: script validation aligns on word
# tokens and ignores punctuation, so this drift passes unnoticed upstream
SEG_STRAIGHT_APOSTROPHE = (
    "This book is a work of fiction. Names, characters, places, and incidents "
    "are the product of the author's imagination or are used fictitiously."
)


class TestLocateSpan:
    def test_exact_texts_span_the_batch(self):
        """control: segments that match the source verbatim locate normally."""
        batch = [
            "Copyright 2024 by Daniel Abraham and Ty Franck",
            "Cover design by Lauren Panepinto",
        ]
        start, end = _locate_span(SOURCE, batch, 0)
        span = SOURCE[start:end]
        assert "Copyright 2024" in span
        assert "Cover design by Lauren Panepinto" in span

    def test_punctuation_drift_still_locates(self):
        """a straight-vs-typographic apostrophe must not defeat location.

        _validate_segments aligns on \\w+ tokens and ignores punctuation, so a
        span builder that requires an exact substring match is stricter than
        the check it feeds.
        """
        start, end = _locate_span(SOURCE, [SEG_STRAIGHT_APOSTROPHE], 0)
        assert start == 0
        # a located segment bounds the span; failing to locate degenerates to
        # cursor -> end-of-source, which merely happens to contain the text
        assert end < len(SOURCE), "did not locate; span fell back to whole source"
        assert "fictitiously" in SOURCE[start:end]

    def test_unlocatable_first_segment_does_not_truncate_span(self):
        """the reported bug: a missed first segment skips the span forward.

        when the first batch text fails to locate, start stays equal to cursor
        and is then reassigned to the next segment that does locate, dropping
        the leading source text out of the span entirely.
        """
        batch = [
            SEG_STRAIGHT_APOSTROPHE,
            "Copyright 2024 by Daniel Abraham and Ty Franck",
        ]
        start, end = _locate_span(SOURCE, batch, 0)
        assert start == 0, f"span skipped the first {start} chars of source"
        assert "work of fiction" in SOURCE[start:end]

    def test_genuinely_absent_first_segment_keeps_cursor(self):
        """text that is nowhere in the source must not advance the start."""
        batch = [
            "An invented sentence appearing nowhere in the source at all.",
            "Cover design by Lauren Panepinto",
        ]
        start, end = _locate_span(SOURCE, batch, 0)
        assert start == 0
        assert "Cover design by Lauren Panepinto" in SOURCE[start:end]

    def test_cursor_is_respected_for_later_batches(self):
        """a later batch starts no earlier than where the previous one ended."""
        cursor = SOURCE.index("Copyright 2024")
        start, end = _locate_span(SOURCE, ["Cover design by Lauren Panepinto"], cursor)
        assert start >= cursor
        assert "Cover design by Lauren Panepinto" in SOURCE[start:end]
