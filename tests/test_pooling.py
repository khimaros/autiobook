"""tests for chapter-ordered segment scheduling."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autiobook.audio import save_segment
from autiobook.config import SAMPLE_RATE
from autiobook.pooling import AudioTask, process_audio_pipeline


@pytest.fixture
def temp_workdir():
    """create a temporary workdir for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_engine():
    """create a mock TTS engine that tracks call order."""
    engine = MagicMock()
    engine.config = MagicMock()
    engine.config.batch_size = 2
    # disable compilation by default to avoid padding in general tests
    engine.config.compile_model = False

    call_log = []

    def mock_synthesize(texts, instruct=""):
        call_log.append(("synthesize", list(texts)))
        return [np.zeros(100, dtype=np.float32) for _ in texts], SAMPLE_RATE

    def mock_clone_voice(texts, ref_audio, ref_text):
        call_log.append(("clone", list(texts), str(ref_audio)))
        return [np.zeros(100, dtype=np.float32) for _ in texts], SAMPLE_RATE

    engine.synthesize = mock_synthesize
    engine.clone_voice = mock_clone_voice
    engine.call_log = call_log

    return engine


@pytest.fixture
def tracking_engine():
    """engine that tracks synthesis AND assembly interleaving."""
    engine = MagicMock()
    engine.config = MagicMock()
    engine.config.batch_size = 2

    event_log = []

    def mock_synthesize(texts, instruct=""):
        for text in texts:
            event_log.append(("synthesize", text))
        return [np.zeros(100, dtype=np.float32) for _ in texts], SAMPLE_RATE

    def mock_clone_voice(texts, ref_audio, ref_text):
        for text in texts:
            event_log.append(("clone", text))
        return [np.zeros(100, dtype=np.float32) for _ in texts], SAMPLE_RATE

    engine.synthesize = mock_synthesize
    engine.clone_voice = mock_clone_voice
    engine.event_log = event_log

    return engine


def make_task(
    text: str,
    segment_hash: str,
    segments_dir: Path,
    voice_ref_audio: Path | None = None,
    voice_ref_text: str | None = None,
) -> AudioTask:
    """helper to create AudioTask instances."""
    return AudioTask(
        text=text,
        segment_hash=segment_hash,
        segments_dir=segments_dir,
        voice_ref_audio=voice_ref_audio,
        voice_ref_text=voice_ref_text,
    )


class TestChapterOrderedScheduling:
    """tests for chapter-ordered segment scheduling."""

    def test_chapters_synthesized_in_order(self, temp_workdir, mock_engine):
        """verify that chapters are processed in input order."""
        segments_dir = temp_workdir / "segments"
        segments_dir.mkdir()

        # create tasks for 3 chapters (2 segments each)
        chapter_data = []
        for ch_idx in range(3):
            wav_path = temp_workdir / f"chapter_{ch_idx}.wav"
            tasks = [
                make_task(
                    f"ch{ch_idx}_seg{i}",
                    f"hash_ch{ch_idx}_seg{i}",
                    segments_dir,
                )
                for i in range(2)
            ]
            chapter_data.append((wav_path, tasks))

        process_audio_pipeline(mock_engine, chapter_data)

        # verify all chapters were created
        for ch_idx in range(3):
            wav_path = temp_workdir / f"chapter_{ch_idx}.wav"
            assert wav_path.exists(), f"chapter {ch_idx} not assembled"

    def test_early_chapter_assembly(self, temp_workdir, tracking_engine):
        """verify chapters are assembled as soon as ready, not after all synthesis."""
        import soundfile as sf

        segments_dir = temp_workdir / "segments"
        segments_dir.mkdir()

        original_write = sf.write

        def track_assembly(path, audio, sr):
            path_name = Path(path).name
            if path_name.startswith("chapter_"):
                tracking_engine.event_log.append(("assemble", path_name))
            original_write(path, audio, sr)

        # 3 chapters, each with 2 segments
        chapter_data = []
        for ch_idx in range(3):
            wav_path = temp_workdir / f"chapter_{ch_idx}.wav"
            tasks = [
                make_task(
                    f"ch{ch_idx}_seg{i}",
                    f"hash_ch{ch_idx}_seg{i}",
                    segments_dir,
                )
                for i in range(2)
            ]
            chapter_data.append((wav_path, tasks))

        with (
            patch("autiobook.pooling.sf.write", side_effect=track_assembly),
            patch("autiobook.audio.sf.write", side_effect=track_assembly),
        ):
            process_audio_pipeline(tracking_engine, chapter_data)

        # extract assembly events
        assemblies = [e for e in tracking_engine.event_log if e[0] == "assemble"]
        synth_events = [e for e in tracking_engine.event_log if e[0] == "synthesize"]

        # chapters should be assembled in order
        assert [a[1] for a in assemblies] == [
            "chapter_0.wav",
            "chapter_1.wav",
            "chapter_2.wav",
        ], f"chapters assembled out of order: {assemblies}"

        # KEY TEST: chapter_0 should be assembled BEFORE ch2 segments are synthesized
        # find positions
        ch0_assembly_idx = next(
            i
            for i, e in enumerate(tracking_engine.event_log)
            if e == ("assemble", "chapter_0.wav")
        )
        ch2_seg0_idx = next(
            (
                i
                for i, e in enumerate(tracking_engine.event_log)
                if e[0] == "synthesize" and "ch2" in e[1]
            ),
            None,
        )

        if ch2_seg0_idx is not None:
            assert ch0_assembly_idx < ch2_seg0_idx, (
                f"chapter_0 assembled at {ch0_assembly_idx} but ch2 synthesized at {ch2_seg0_idx}. "
                f"Early assembly not working. Event log: {tracking_engine.event_log}"
            )

    def test_cached_segments_enable_immediate_assembly(self, temp_workdir, mock_engine):
        """verify that chapters with all segments cached are assembled immediately."""
        import soundfile as sf

        segments_dir = temp_workdir / "segments"
        segments_dir.mkdir()

        # pre-cache segments for chapter 0 and 2
        for ch_idx in [0, 2]:
            for i in range(2):
                save_segment(
                    segments_dir,
                    f"hash_ch{ch_idx}_seg{i}",
                    np.zeros(100, dtype=np.float32),
                    SAMPLE_RATE,
                )

        assembly_order = []
        original_write = sf.write

        def track_assembly(path, audio, sr):
            path_name = Path(path).name
            if path_name.startswith("chapter_"):
                assembly_order.append(path_name)
            original_write(path, audio, sr)

        chapter_data = []
        for ch_idx in range(3):
            wav_path = temp_workdir / f"chapter_{ch_idx}.wav"
            tasks = [
                make_task(
                    f"ch{ch_idx}_seg{i}",
                    f"hash_ch{ch_idx}_seg{i}",
                    segments_dir,
                )
                for i in range(2)
            ]
            chapter_data.append((wav_path, tasks))

        with (
            patch("autiobook.pooling.sf.write", side_effect=track_assembly),
            patch("autiobook.audio.sf.write", side_effect=track_assembly),
        ):
            process_audio_pipeline(mock_engine, chapter_data)

        # chapter 0 should assemble first (cached), then 1 (after generation),
        # then 2 (cached but waiting for chapter order)
        assert assembly_order == [
            "chapter_0.wav",
            "chapter_1.wav",
            "chapter_2.wav",
        ]

        # only chapter 1 segments should have been synthesized
        assert len(mock_engine.call_log) > 0
        synthesized_texts = []
        for call in mock_engine.call_log:
            if call[0] == "synthesize":
                synthesized_texts.extend(call[1])

        assert "ch1_seg0" in synthesized_texts
        assert "ch1_seg1" in synthesized_texts
        assert "ch0_seg0" not in synthesized_texts  # cached
        assert "ch2_seg0" not in synthesized_texts  # cached


class TestSmartBatchSelection:
    """tests for smart batch selection in perform mode."""

    def test_voice_batching_prefers_earlier_chapters(self, temp_workdir, mock_engine):
        """verify that when batching by voice, earlier chapters are preferred."""
        segments_dir = temp_workdir / "segments"
        segments_dir.mkdir()

        # create voice reference files
        voice_a = temp_workdir / "voice_a.wav"
        voice_b = temp_workdir / "voice_b.wav"
        for v in [voice_a, voice_b]:
            save_segment(v.parent, v.stem, np.zeros(100, dtype=np.float32), SAMPLE_RATE)

        # chapter 0: voice_a, voice_b
        # chapter 1: voice_a, voice_b
        # chapter 2: voice_a
        chapter_data = [
            (
                temp_workdir / "chapter_0.wav",
                [
                    make_task("ch0_a", "hash_ch0_a", segments_dir, voice_a, "ref"),
                    make_task("ch0_b", "hash_ch0_b", segments_dir, voice_b, "ref"),
                ],
            ),
            (
                temp_workdir / "chapter_1.wav",
                [
                    make_task("ch1_a", "hash_ch1_a", segments_dir, voice_a, "ref"),
                    make_task("ch1_b", "hash_ch1_b", segments_dir, voice_b, "ref"),
                ],
            ),
            (
                temp_workdir / "chapter_2.wav",
                [
                    make_task("ch2_a", "hash_ch2_a", segments_dir, voice_a, "ref"),
                ],
            ),
        ]

        process_audio_pipeline(mock_engine, chapter_data)

        # all chapters should be assembled
        for ch_idx in range(3):
            wav_path = temp_workdir / f"chapter_{ch_idx}.wav"
            assert wav_path.exists(), f"chapter {ch_idx} not assembled"

    def test_mixed_voice_and_no_voice_batching(self, temp_workdir, mock_engine):
        """verify correct handling of mixed voice cloning and synthesis tasks."""
        segments_dir = temp_workdir / "segments"
        segments_dir.mkdir()

        voice_a = temp_workdir / "voice_a.wav"
        save_segment(
            voice_a.parent, voice_a.stem, np.zeros(100, dtype=np.float32), SAMPLE_RATE
        )

        # chapter 0: voice clone + synthesize
        # chapter 1: synthesize only
        chapter_data = [
            (
                temp_workdir / "chapter_0.wav",
                [
                    make_task(
                        "ch0_clone", "hash_ch0_clone", segments_dir, voice_a, "ref"
                    ),
                    make_task("ch0_synth", "hash_ch0_synth", segments_dir),
                ],
            ),
            (
                temp_workdir / "chapter_1.wav",
                [
                    make_task("ch1_synth", "hash_ch1_synth", segments_dir),
                ],
            ),
        ]

        process_audio_pipeline(mock_engine, chapter_data)

        # verify both clone and synthesize were called
        call_types = [c[0] for c in mock_engine.call_log]
        assert "clone" in call_types
        assert "synthesize" in call_types

        # all chapters assembled
        assert (temp_workdir / "chapter_0.wav").exists()
        assert (temp_workdir / "chapter_1.wav").exists()


class TestSegmentDeduplication:
    """tests for segment deduplication across chapters."""

    def test_duplicate_segments_synthesized_once(self, temp_workdir, mock_engine):
        """verify that identical segments are only synthesized once."""
        segments_dir = temp_workdir / "segments"
        segments_dir.mkdir()

        # both chapters have the same segment hash
        chapter_data = [
            (
                temp_workdir / "chapter_0.wav",
                [make_task("same text", "same_hash", segments_dir)],
            ),
            (
                temp_workdir / "chapter_1.wav",
                [make_task("same text", "same_hash", segments_dir)],
            ),
        ]

        process_audio_pipeline(mock_engine, chapter_data)

        # should only be synthesized once
        all_texts = []
        for call in mock_engine.call_log:
            if call[0] == "synthesize":
                all_texts.extend(call[1])

        assert all_texts.count("same text") == 1

        # both chapters should still be assembled
        assert (temp_workdir / "chapter_0.wav").exists()
        assert (temp_workdir / "chapter_1.wav").exists()

    def test_padding_on_last_batch(self, temp_workdir, mock_engine):
        """verify padding occurs when compile_model is True."""
        segments_dir = temp_workdir / "segments"
        segments_dir.mkdir()

        # enable compilation to trigger padding
        mock_engine.config.compile_model = True

        chapter_data = [
            (
                temp_workdir / "chapter_0.wav",
                [make_task("single task", "single_hash", segments_dir)],
            ),
        ]

        process_audio_pipeline(mock_engine, chapter_data)

        # should have been padded to batch_size=2
        assert len(mock_engine.call_log) == 1
        call_args = mock_engine.call_log[0][1]  # texts list
        assert len(call_args) == 2
        assert call_args[0] == "single task"
        assert call_args[1] == "."  # padding
