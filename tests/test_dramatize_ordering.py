"""tests for how dramatize_book orders phases across chapters.

the script phase keys its resume hash on the whole cast, so a character
discovered while casting chapter N invalidates the scripts of chapters
1..N-1. casting every chapter before anything reads the cast is what keeps a
run converging.
"""

from unittest.mock import patch

import pytest

from autiobook.dramatize import dramatize_book

CHAPTERS = [1, 2, 3]


@pytest.fixture
def workdir(tmp_path):
    extract = tmp_path / "extract"
    extract.mkdir(parents=True)
    for num, title in ((1, "Contents"), (2, "Chapter_2"), (3, "PROLOGUE")):
        (extract / f"{num:02d}_{title}.txt").write_text("text", encoding="utf-8")
    return tmp_path


class Recorder:
    """records (phase, chapters) in call order across the patched phases."""

    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def _record(self, phase, index):
        def fn(*args, **kwargs):
            chapters = args[index] if len(args) > index else kwargs.get("chapters")
            self.calls.append((phase, chapters))

        return fn

    def phases(self):
        return [p for p, _ in self.calls]

    def chapters_for(self, phase):
        return [c for p, c in self.calls if p == phase]


def _run(workdir, recorder, **kwargs):
    with patch("autiobook.dramatize.run_cast_generation", recorder._record("cast", 4)):
        with patch("autiobook.audition.run_audition", recorder._record("audition", 99)):
            with patch(
                "autiobook.dramatize._run_script_phases", recorder._record("script", 1)
            ):
                with patch("autiobook.dramatize._check_unresolved_flags"):
                    with patch(
                        "autiobook.dramatize._run_perform_phases",
                        recorder._record("perform", 1),
                    ):
                        dramatize_book(workdir, chapters=CHAPTERS, **kwargs)
    return recorder


class TestCastRunsAcrossEveryChapterFirst:
    def test_phase_wise_is_the_default(self, workdir):
        r = _run(workdir, Recorder())

        assert r.phases() == ["cast", "audition", "script", "perform"]
        assert r.chapters_for("script") == [CHAPTERS]

    def test_chapter_wise_still_casts_all_chapters_up_front(self, workdir):
        r = _run(workdir, Recorder(), chapter_wise=True)

        # one cast pass covering every chapter, before any script work
        assert r.chapters_for("cast") == [CHAPTERS]
        assert r.phases()[:2] == ["cast", "audition"]

    def test_chapter_wise_runs_the_tail_per_chapter(self, workdir):
        r = _run(workdir, Recorder(), chapter_wise=True)

        assert r.chapters_for("script") == [[1], [2], [3]]
        assert r.chapters_for("perform") == [[1], [2], [3]]

    def test_voice_phases_run_once_not_per_chapter(self, workdir):
        """audition and emote were never chapter-scoped; calling them per
        chapter only reprinted a resume no-op."""
        r = _run(workdir, Recorder(), chapter_wise=True)

        assert r.phases().count("cast") == 1
        assert r.phases().count("audition") == 1

    def test_cast_precedes_every_script_call(self, workdir):
        for kwargs in ({}, {"chapter_wise": True}):
            phases = _run(workdir, Recorder(), **kwargs).phases()
            assert phases.index("cast") < phases.index("script")
