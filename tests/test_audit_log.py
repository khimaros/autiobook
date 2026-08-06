"""tests for audit log merging across pipeline phases.

all phases share workdir/audit/audit.json. review is the only writer that owns
a whole set of entries at once, so it must replace its own prior findings for
the chapters it just processed without discarding anything else -- cast merges,
revise flags, or other chapters' results.
"""

import json

from autiobook.dramatize import (
    REVIEW_AUDIT_PHASE,
    _load_audit,
    _save_audit,
    _save_review_audit,
)


def entry(kind, chapter, phase=None, **extra):
    e = {"kind": kind, "chapter": chapter, **extra}
    if phase:
        e["phase"] = phase
    return e


def review_entry(kind, chapter, **extra):
    return entry(kind, chapter, phase=REVIEW_AUDIT_PHASE, **extra)


class TestSaveReviewAudit:
    def test_preserves_other_chapters(self, tmp_path):
        """reviewing chapter 5 must not erase chapter 3's findings."""
        path = tmp_path / "audit.json"
        _save_audit(path, [review_entry("flag", "03_PART_ONE", segment=7)])

        _save_review_audit(
            path, [review_entry("edit", "05_Two", segment=98)], {"05_Two"}
        )

        kept = _load_audit(path)
        assert {e["chapter"] for e in kept} == {"03_PART_ONE", "05_Two"}

    def test_preserves_cast_merge_entries(self, tmp_path):
        """cast merges are written by a different phase and must survive."""
        path = tmp_path / "audit.json"
        _save_audit(path, [entry("cast_merge", "04_One", merged="Nol->Noel")])

        _save_review_audit(
            path, [review_entry("flag", "04_One", segment=1)], {"04_One"}
        )

        kinds = [e["kind"] for e in _load_audit(path)]
        assert "cast_merge" in kinds
        assert "flag" in kinds

    def test_preserves_revise_flag_for_same_chapter(self, tmp_path):
        """revise flags quote-structure problems review never inspects.

        both write kind="flag", so the phase tag is what keeps review from
        clearing a flag it did not author and silently un-gating perform.
        """
        path = tmp_path / "audit.json"
        _save_audit(path, [entry("flag", "05_Two", segment=12, reason="mixed quotes")])

        _save_review_audit(
            path, [review_entry("edit", "05_Two", segment=98)], {"05_Two"}
        )

        kept = _load_audit(path)
        assert any(
            e.get("reason") == "mixed quotes" for e in kept
        ), "revise flag was discarded by review"

    def test_replaces_its_own_prior_entries(self, tmp_path):
        """re-reviewing a chapter refreshes rather than duplicates."""
        path = tmp_path / "audit.json"
        _save_audit(path, [review_entry("validation", "05_Two", batch=1)])

        _save_review_audit(
            path, [review_entry("edit", "05_Two", segment=98)], {"05_Two"}
        )

        kept = _load_audit(path)
        assert len(kept) == 1
        assert kept[0]["kind"] == "edit"

    def test_repeated_saves_do_not_duplicate(self, tmp_path):
        """review saves after every batch; the merge must be idempotent."""
        path = tmp_path / "audit.json"
        audit = [review_entry("edit", "05_Two", segment=98)]

        for _ in range(3):
            _save_review_audit(path, audit, {"05_Two"})

        assert len(_load_audit(path)) == 1

    def test_accumulates_across_chapters_in_one_run(self, tmp_path):
        """chapter-wise review appends each chapter as it completes."""
        path = tmp_path / "audit.json"
        audit = [review_entry("edit", "04_One", segment=3)]
        _save_review_audit(path, audit, {"04_One"})

        audit.append(review_entry("flag", "05_Two", segment=51))
        _save_review_audit(path, audit, {"04_One", "05_Two"})

        assert {e["chapter"] for e in _load_audit(path)} == {"04_One", "05_Two"}

    def test_clears_stale_entries_when_chapter_now_clean(self, tmp_path):
        """a chapter that re-reviews with no findings loses its old entries."""
        path = tmp_path / "audit.json"
        _save_audit(path, [review_entry("validation", "05_Two", batch=1)])

        _save_review_audit(path, [], {"05_Two"})

        assert _load_audit(path) == []

    def test_missing_file_is_treated_as_empty(self, tmp_path):
        path = tmp_path / "audit.json"
        _save_review_audit(
            path, [review_entry("edit", "05_Two", segment=1)], {"05_Two"}
        )
        assert len(json.loads(path.read_text())) == 1
