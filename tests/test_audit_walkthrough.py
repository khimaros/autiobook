"""tests for the interactive audit walkthrough.

an open flag defers its chapter from perform, so which key clears an entry and
which merely advances is the difference between a chapter shipping and being
silently skipped.
"""

import argparse
import json
from unittest.mock import patch

from autiobook.dramatize import _audit_path, cmd_audit


def _entry(chapter: str = "05_CHAPTER", segment: int = 2, suggested=None) -> dict:
    e = {
        "kind": "flag",
        "phase": "review",
        "chapter": chapter,
        "segment": segment,
        "reason": "speaker is ambiguous",
        "text": "some narration",
    }
    if suggested:
        e["suggested"] = suggested
    return e


def _seed(workdir, entries: list[dict]):
    path = _audit_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))
    return path


def _args(workdir):
    return argparse.Namespace(
        workdir=str(workdir),
        all=False,
        clear=False,
        list=False,
        api_base="http://localhost:8080/v1",
        api_key="",
        model="test-model",
        review_model=None,
        thinking_budget=0,
    )


def _walk(workdir, keys: list[str]) -> list[dict]:
    """drive the walkthrough with scripted keypresses; returns entries left."""
    with patch("builtins.input", side_effect=keys):
        cmd_audit(_args(workdir))
    return json.loads(_audit_path(workdir).read_text())


class TestDismiss:
    def test_dismiss_clears_the_entry(self, tmp_path):
        _seed(tmp_path, [_entry()])
        assert _walk(tmp_path, ["d"]) == []

    def test_dismiss_clears_with_a_suggestion_present(self, tmp_path):
        """dismissing declines the suggestion; it must still resolve the flag."""
        _seed(tmp_path, [_entry(suggested={"speaker": "Kern"})])
        assert _walk(tmp_path, ["d"]) == []

    def test_long_form_dismiss(self, tmp_path):
        _seed(tmp_path, [_entry()])
        assert _walk(tmp_path, ["dismiss"]) == []

    def test_dismiss_only_clears_the_current_entry(self, tmp_path):
        _seed(tmp_path, [_entry(segment=1), _entry(segment=2)])
        left = _walk(tmp_path, ["d", "q"])
        assert [e["segment"] for e in left] == [2]


class TestNext:
    def test_next_leaves_the_entry_open(self, tmp_path):
        _seed(tmp_path, [_entry()])
        left = _walk(tmp_path, ["n"])
        assert [e["segment"] for e in left] == [2]

    def test_next_advances_through_every_entry(self, tmp_path):
        _seed(tmp_path, [_entry(segment=1), _entry(segment=2)])
        left = _walk(tmp_path, ["n", "n"])
        assert [e["segment"] for e in left] == [1, 2]

    def test_quit_leaves_the_rest_open(self, tmp_path):
        _seed(tmp_path, [_entry(segment=1), _entry(segment=2)])
        left = _walk(tmp_path, ["q"])
        assert [e["segment"] for e in left] == [1, 2]


class TestApplySuggestion:
    def test_apply_writes_and_clears(self, tmp_path):
        _seed(tmp_path, [_entry(suggested={"speaker": "Kern"})])
        with patch(
            "autiobook.dramatize._apply_flag_suggestion", return_value=True
        ) as applied:
            left = _walk(tmp_path, ["a"])
        applied.assert_called_once()
        assert left == []

    def test_apply_without_a_suggestion_just_advances(self, tmp_path):
        """a is not offered on a bare flag and must not clear it."""
        _seed(tmp_path, [_entry()])
        left = _walk(tmp_path, ["a"])
        assert [e["segment"] for e in left] == [2]

    def test_failed_apply_keeps_the_entry(self, tmp_path):
        _seed(tmp_path, [_entry(suggested={"speaker": "Kern"})])
        with patch("autiobook.dramatize._apply_flag_suggestion", return_value=False):
            left = _walk(tmp_path, ["a", "q"])
        assert len(left) == 1


class TestSuggest:
    """most flags arrive with no suggestion; [s]uggest asks the llm for one."""

    def _chapter(self, workdir, speaker="Narrator"):
        """write a script and its source so a flagged segment can be reviewed."""
        from autiobook.dramatize import save_script
        from autiobook.llm import ScriptSegment
        from autiobook.resume import get_command_dir

        segs = [
            ScriptSegment(speaker=speaker, text=f"line {i}.", instruction="neutral")
            for i in range(5)
        ]
        save_script(
            get_command_dir(workdir, "script") / "05_CHAPTER.json",
            segs,
        )
        (get_command_dir(workdir, "extract") / "05_CHAPTER.txt").write_text(
            " ".join(s.text for s in segs)
        )
        return segs

    def test_suggest_records_a_correction(self, tmp_path):
        from autiobook.llm import ScriptSegment

        segs = self._chapter(tmp_path)
        _seed(tmp_path, [_entry(chapter="05_CHAPTER", segment=2)])
        corrected = list(segs[:5])
        # the reviewer reattributes the flagged segment (index 1 of the window)
        corrected[1] = ScriptSegment(
            speaker="Kern", text=corrected[1].text, instruction="stern"
        )

        with patch(
            "autiobook.dramatize.review_script_batch",
            return_value=(corrected, [], [], [], []),
        ):
            left = _walk(tmp_path, ["s", "q"])

        assert left[0]["suggested"] == {"speaker": "Kern", "instruction": "stern"}

    def test_no_change_leaves_the_entry_alone(self, tmp_path):
        segs = self._chapter(tmp_path)
        _seed(tmp_path, [_entry(chapter="05_CHAPTER", segment=2)])

        with patch(
            "autiobook.dramatize.review_script_batch",
            return_value=(list(segs), [], [], [], []),
        ):
            left = _walk(tmp_path, ["s", "q"])

        assert "suggested" not in left[0]

    def test_llm_failure_keeps_the_entry(self, tmp_path):
        self._chapter(tmp_path)
        _seed(tmp_path, [_entry(chapter="05_CHAPTER", segment=2)])

        with patch(
            "autiobook.dramatize.review_script_batch",
            side_effect=RuntimeError("llm http 500"),
        ):
            left = _walk(tmp_path, ["s", "q"])

        assert len(left) == 1
        assert "suggested" not in left[0]

    def test_suggest_is_not_offered_once_one_exists(self, tmp_path):
        _seed(tmp_path, [_entry(suggested={"speaker": "Kern"})])
        seen: list[str] = []

        def capture(msg):
            seen.append(msg)
            return "q"

        with patch("builtins.input", side_effect=capture):
            cmd_audit(_args(tmp_path))
        assert "[s]uggest" not in seen[0]
        assert "[a]pply" in seen[0]

    def test_suggest_offered_on_a_bare_flag(self, tmp_path):
        _seed(tmp_path, [_entry()])
        seen: list[str] = []

        def capture(msg):
            seen.append(msg)
            return "q"

        with patch("builtins.input", side_effect=capture):
            cmd_audit(_args(tmp_path))
        assert "[s]uggest" in seen[0]


class TestSegmentDrift:
    """revise's splitting renumbers segments, so a recorded number goes stale.

    the recorded text is the stable identity; trusting the number would apply
    corrections to whatever line happens to sit at that index now.
    """

    def _script(self, workdir, texts, speaker="Narrator"):
        from autiobook.dramatize import save_script
        from autiobook.llm import ScriptSegment
        from autiobook.resume import get_command_dir

        segs = [
            ScriptSegment(speaker=speaker, text=t, instruction="neutral") for t in texts
        ]
        path = get_command_dir(workdir, "script") / "05_CHAPTER.json"
        save_script(path, segs)
        (get_command_dir(workdir, "extract") / "05_CHAPTER.txt").write_text(
            " ".join(texts)
        )
        return path

    def _entry_for(self, text, segment):
        e = _entry(chapter="05_CHAPTER", segment=segment)
        e["text"] = text
        return e

    def test_index_resolves_by_text_when_it_drifted(self, tmp_path):
        from autiobook.dramatize import _audit_segment_index, load_script

        path = self._script(tmp_path, ["a.", "b.", "c.", "target.", "e."])
        segments = load_script(path)
        # the entry was written when "target." was segment 2
        assert _audit_segment_index(self._entry_for("target.", 2), segments) == 3

    def test_exact_index_is_kept_when_text_still_matches(self, tmp_path):
        from autiobook.dramatize import _audit_segment_index, load_script

        path = self._script(tmp_path, ["a.", "b.", "c."])
        segments = load_script(path)
        assert _audit_segment_index(self._entry_for("b.", 2), segments) == 1

    def test_ambiguous_text_falls_back_to_the_number(self, tmp_path):
        """two identical segments give no evidence either way."""
        from autiobook.dramatize import _audit_segment_index, load_script

        path = self._script(tmp_path, ["dup.", "x.", "dup."])
        segments = load_script(path)
        assert _audit_segment_index(self._entry_for("dup.", 3), segments) == 2

    def test_missing_text_falls_back_to_the_number(self, tmp_path):
        from autiobook.dramatize import _audit_segment_index, load_script

        path = self._script(tmp_path, ["a.", "b.", "c."])
        segments = load_script(path)
        assert _audit_segment_index(self._entry_for("gone.", 2), segments) == 1

    def test_out_of_range_number_with_no_text_match(self, tmp_path):
        from autiobook.dramatize import _audit_segment_index, load_script

        path = self._script(tmp_path, ["a."])
        segments = load_script(path)
        assert _audit_segment_index(self._entry_for("gone.", 99), segments) is None

    def test_apply_writes_to_the_resolved_segment(self, tmp_path):
        from autiobook.dramatize import load_script

        path = self._script(tmp_path, ["a.", "b.", "c.", "target.", "e."])
        e = self._entry_for("target.", 2)
        e["suggested"] = {"speaker": "Kern", "instruction": "stern"}
        _seed(tmp_path, [e])

        assert _walk(tmp_path, ["a"]) == []

        segments = load_script(path)
        assert segments[3].speaker == "Kern"
        assert segments[1].speaker == "Narrator"  # the stale index is untouched

    def test_window_reports_the_drift(self, tmp_path):
        from autiobook.dramatize import (
            _audit_segment_window,
            _format_segment_window,
        )

        self._script(tmp_path, ["a.", "b.", "c.", "target.", "e."])
        e = self._entry_for("target.", 2)
        lines = _format_segment_window(
            _audit_segment_window(tmp_path, e), recorded_no=e["segment"]
        )
        joined = "\n".join(lines)
        assert "> " in joined and "target." in joined
        assert "recorded as seg 2" in joined


class TestClosedStdin:
    """a walkthrough that cannot read a choice must end, not raise."""

    def test_eof_ends_the_walkthrough(self, tmp_path):
        _seed(tmp_path, [_entry(segment=1), _entry(segment=2)])
        with patch("builtins.input", side_effect=EOFError):
            cmd_audit(_args(tmp_path))
        left = json.loads(_audit_path(tmp_path).read_text())
        assert [e["segment"] for e in left] == [1, 2]

    def test_interrupt_ends_the_walkthrough(self, tmp_path):
        _seed(tmp_path, [_entry()])
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            cmd_audit(_args(tmp_path))

    def test_prompt_choice_reads_eof_as_quit(self):
        from autiobook.utils import prompt_choice

        with patch("builtins.input", side_effect=EOFError):
            assert prompt_choice("> ") == "q"
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert prompt_choice("> ") == "q"


class TestPrompt:
    """the key set is offered without duplicates."""

    def _prompt_for(self, tmp_path, entry) -> str:
        _seed(tmp_path, [entry])
        seen: list[str] = []

        def capture(msg):
            seen.append(msg)
            return "q"

        with patch("builtins.input", side_effect=capture):
            cmd_audit(_args(tmp_path))
        return seen[0]

    FIXED = "[p]ager / [e]dit / [d]ismiss / [n]ext / [q]uit>"

    def test_bare_flag_offers_suggest_not_apply(self, tmp_path):
        prompt = self._prompt_for(tmp_path, _entry())
        assert prompt.strip() == f"[s]uggest / {self.FIXED}"

    def test_suggestion_puts_apply_first(self, tmp_path):
        """the conditional options lead, so the rest never shift position."""
        prompt = self._prompt_for(tmp_path, _entry(suggested={"speaker": "Kern"}))
        assert prompt.strip() == f"[a]pply / {self.FIXED}"

    def test_non_flag_entry_offers_neither(self, tmp_path):
        """a validation record has no segment attribution to suggest for."""
        entry = _entry()
        entry["kind"] = "validation"
        prompt = self._prompt_for(tmp_path, entry)
        assert prompt.strip() == self.FIXED

    def test_keep_is_gone(self, tmp_path):
        """keep was a second key for what next already did."""
        prompt = self._prompt_for(tmp_path, _entry())
        assert "[k]eep" not in prompt
