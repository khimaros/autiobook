"""tests for retained-segment protection and tts text normalization.

both failures showed up on the same front matter: review promoted a blurb
attribution from "Retained" to "Narrator", and the tts backend then answered
the leading em dash with silence, which retake spent its whole budget on.
"""

from unittest.mock import patch

import numpy as np

from autiobook.config import RETAINED_SPEAKERS
from autiobook.llm import (
    Character,
    ScriptSegment,
    _apply_review_changes,
    _format_cast_list,
)
from autiobook.utils import normalize_tts_text

ATTRIBUTION = "—Financial Times"


def _seg(speaker: str, text: str = "x", instruction: str = "neutral"):
    return ScriptSegment(speaker=speaker, text=text, instruction=instruction)


class TestRetainedProtection:
    """review may not voice text marked as not-to-be-spoken."""

    def test_retained_to_narrator_is_rejected(self):
        original = [_seg("Retained", ATTRIBUTION)]
        changes = [{"index": 0, "speaker": "Narrator", "instruction": "neutral"}]

        merged, _muts, _instr, retained = _apply_review_changes(original, changes)

        assert merged[0].speaker == "Retained"
        assert retained == [(0, "Narrator")]

    def test_every_retained_speaker_is_protected(self):
        for name in sorted(RETAINED_SPEAKERS):
            merged, _m, _i, retained = _apply_review_changes(
                [_seg(name)], [{"index": 0, "speaker": "Narrator"}]
            )
            assert merged[0].speaker == name, name
            assert retained == [(0, "Narrator")], name

    def test_retained_to_retained_is_allowed(self):
        """normalizing between retained variants is not a promotion to speech."""
        merged, _m, _i, retained = _apply_review_changes(
            [_seg("Silent")], [{"index": 0, "speaker": "Retained"}]
        )
        assert merged[0].speaker == "Retained"
        assert retained == []

    def test_promoting_narration_to_retained_is_allowed(self):
        """the guard is one-way: review may still silence a chapter number."""
        merged, _m, _i, retained = _apply_review_changes(
            [_seg("Narrator", "[iv]")], [{"index": 0, "speaker": "Retained"}]
        )
        assert merged[0].speaker == "Retained"
        assert retained == []

    def test_instruction_change_on_retained_segment_still_applies(self):
        merged, _m, _i, retained = _apply_review_changes(
            [_seg("Retained")], [{"index": 0, "instruction": "sad"}]
        )
        assert merged[0].speaker == "Retained"
        assert merged[0].instruction == "sad"
        assert retained == []

    def test_ordinary_speaker_correction_is_untouched(self):
        merged, _m, _i, retained = _apply_review_changes(
            [_seg("Narrator")], [{"index": 0, "speaker": "Extra Male"}]
        )
        assert merged[0].speaker == "Extra Male"
        assert retained == []


class TestReviewCastList:
    """review is told to use names exactly as listed, so the list must hold
    every name it is expected to preserve."""

    def _cast(self):
        return [Character(name="Portia", description="d")]

    def test_specials_present_for_review(self):
        listed = _format_cast_list(self._cast(), specials=True)
        for name in ["Portia", "Narrator", "Extra Female", *RETAINED_SPEAKERS]:
            assert name in listed, name

    def test_generation_list_stays_characters_only(self):
        listed = _format_cast_list(self._cast())
        assert "Portia" in listed
        assert "Retained" not in listed


class TestNormalizeTTSText:
    """the script keeps its dash; the synthesis request does not."""

    def test_em_dash_attribution_is_stripped(self):
        assert normalize_tts_text(ATTRIBUTION) == "Financial Times"

    def test_every_dash_form_is_stripped(self):
        for dash in ["-", "--", "‐", "–", "—", "―", "−"]:
            assert normalize_tts_text(f"{dash} SFBook") == "SFBook", dash

    def test_interior_dashes_are_kept(self):
        """dashes inside a line shape prosody and must survive."""
        line = "big themes—gods, messiahs—with brio"
        assert normalize_tts_text(line) == line

    def test_dash_only_line_is_left_alone(self):
        """never turn a segment into an empty synthesis request."""
        assert normalize_tts_text("———") == "———"

    def test_ordinary_text_untouched(self):
        assert normalize_tts_text("Portia waits.") == "Portia waits."


class FakeEngine:
    """records the text it was asked to speak."""

    seeded = False

    def __init__(self):
        self.config = type("cfg", (), {"seed": 42, "batch_size": 1})()
        self.requested: list[list[str]] = []

    def synthesize(self, texts, instruct="", speaker=None):
        self.requested.append(list(texts))
        return [np.zeros(100, dtype=np.float32) for _ in texts], 24000


class TestSynthesisUsesNormalizedText:
    def test_leading_dash_stripped_before_the_request(self, tmp_path):
        from autiobook.pooling import AudioTask, _run_synthesis

        engine = FakeEngine()
        task = AudioTask(
            text=ATTRIBUTION,
            segment_hash="abc123",
            segments_dir=tmp_path,
            preset_voice="Umbriel",
        )
        _run_synthesis(engine, [task])

        assert engine.requested == [["Financial Times"]]
        # the task itself is unchanged, so the segment hash stays keyed on the
        # script text and cached takes are not invalidated
        assert task.text == ATTRIBUTION


class TestRetakeSeedReporting:
    """a seed the backend never receives must not appear in the log."""

    def test_unseeded_backend_omits_seed(self, tmp_path):
        from autiobook.pooling import AudioTask, _retry_bad_takes

        engine = FakeEngine()  # seeded = False
        task = AudioTask(
            text="hello", segment_hash="deadbeef" * 2, segments_dir=tmp_path
        )
        lines: list[str] = []
        silent = np.zeros(100, dtype=np.float32)

        with patch("autiobook.pooling.tqdm.write", lines.append):
            try:
                _retry_bad_takes(engine, [task], [silent], max_attempts=1, verbose=True)
            except RuntimeError:
                pass

        assert any("(silent)" in ln for ln in lines)
        assert not any("seed=" in ln for ln in lines)

    def test_seeded_backend_reports_seed(self, tmp_path):
        from autiobook.pooling import AudioTask, _retry_bad_takes

        engine = FakeEngine()
        engine.seeded = True
        task = AudioTask(
            text="hello", segment_hash="deadbeef" * 2, segments_dir=tmp_path
        )
        lines: list[str] = []
        silent = np.zeros(100, dtype=np.float32)

        with patch("autiobook.pooling.tqdm.write", lines.append):
            try:
                _retry_bad_takes(engine, [task], [silent], max_attempts=1, verbose=True)
            except RuntimeError:
                pass

        assert any("seed=" in ln for ln in lines)
