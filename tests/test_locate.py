"""tests for audio-time to segment lookup."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore


def _write_wav(path: Path, duration_s: float, sample_rate: int = 24000) -> None:
    audio = np.zeros(int(duration_s * sample_rate), dtype=np.float32)
    sf.write(str(path), audio, sample_rate)


def _setup_chapter(tmp_path: Path) -> tuple[Path, Path]:
    """build a minimal chapter with 3 chunks of known durations + a script."""
    perform = tmp_path / "perform"
    segments = perform / "segments"
    segments.mkdir(parents=True)
    script = tmp_path / "script" / "01_ch.json"
    script.parent.mkdir()

    durations = [1.0, 2.0, 0.5]
    hashes = ["aaaa", "bbbb", "cccc"]
    for h, d in zip(hashes, durations):
        _write_wav(segments / f"{h}.wav", d)

    script.write_text(
        json.dumps(
            {
                "version": 2,
                "segments": [
                    {"speaker": "Narrator", "text": "first", "instruction": "neutral"},
                    {"speaker": "Alice", "text": "second", "instruction": "happy"},
                    {"speaker": "Narrator", "text": "third", "instruction": "neutral"},
                ],
            }
        )
    )

    wav_path = perform / "01_ch.wav"
    from autiobook.pooling import write_timing_manifest

    chunk_meta = [
        {"script_idx": 0, "chunk_idx": 0, "script_path": str(script)},
        {"script_idx": 1, "chunk_idx": 0, "script_path": str(script)},
        {"script_idx": 2, "chunk_idx": 0, "script_path": str(script)},
    ]
    _write_wav(wav_path, sum(durations) + 2 * 0.5)  # placeholder chapter wav
    write_timing_manifest(wav_path, segments, hashes, chunk_meta)
    return wav_path, segments


class TestLocate:
    def test_locate_first_chunk(self, tmp_path):
        from autiobook.locate import locate_segment

        wav, segs = _setup_chapter(tmp_path)
        loc = locate_segment(wav, 0.5)
        assert loc.chunk_wav == segs / "aaaa.wav"
        assert loc.chunk_hash == "aaaa"
        assert loc.script_idx == 0
        assert loc.speaker == "Narrator"
        assert loc.text == "first"

    def test_locate_middle_chunk_with_pause(self, tmp_path):
        """after chunk 0 (0-1.0s) + pause (1.0-1.5s), chunk 1 starts at 1.5s."""
        from autiobook.locate import locate_segment

        wav, segs = _setup_chapter(tmp_path)
        loc = locate_segment(wav, 2.0)
        assert loc.chunk_wav == segs / "bbbb.wav"
        assert loc.script_idx == 1
        assert loc.speaker == "Alice"
        assert loc.chunk_start_s == pytest.approx(1.5)
        assert loc.chunk_end_s == pytest.approx(3.5)

    def test_locate_last_chunk(self, tmp_path):
        from autiobook.locate import locate_segment

        wav, segs = _setup_chapter(tmp_path)
        loc = locate_segment(wav, 4.2)  # inside last chunk
        assert loc.chunk_hash == "cccc"
        assert loc.script_idx == 2

    def test_out_of_range_raises(self, tmp_path):
        from autiobook.locate import locate_segment

        wav, _ = _setup_chapter(tmp_path)
        with pytest.raises(ValueError, match="out of range"):
            locate_segment(wav, 99.0)

    def test_parse_time_formats(self):
        from autiobook.locate import parse_time

        assert parse_time("42") == 42.0
        assert parse_time("83.5") == 83.5
        assert parse_time("1:23") == 83.0
        assert parse_time("1:23.5") == 83.5

    def test_parse_time_invalid(self):
        from autiobook.locate import parse_time

        with pytest.raises(ValueError):
            parse_time("not-a-time")

    def test_missing_manifest(self, tmp_path):
        from autiobook.locate import locate_segment

        wav = tmp_path / "x.wav"
        _write_wav(wav, 1.0)
        with pytest.raises(FileNotFoundError, match="timing manifest"):
            locate_segment(wav, 0.5)


class TestManifestBackfill:
    """re-running perform/dramatize regenerates a missing timing manifest
    without needing to re-assemble the chapter wav."""

    def test_backfill_when_wav_fresh(self, tmp_path):
        from unittest.mock import MagicMock

        from autiobook.pooling import (
            AudioTask,
            process_audio_pipeline,
            timing_manifest_path,
        )

        # arrange: cached chapter wav + cached chunks, no manifest.
        perform = tmp_path / "perform"
        segments = perform / "segments"
        segments.mkdir(parents=True)
        for h in ("aaaa", "bbbb"):
            _write_wav(segments / f"{h}.wav", 1.0)
        wav_path = perform / "01_ch.wav"
        _write_wav(wav_path, 2.5)

        tasks = [
            AudioTask(
                text="t1",
                segment_hash="aaaa",
                segments_dir=segments,
                metadata={"script_idx": 0, "chunk_idx": 0, "script_path": "x"},
            ),
            AudioTask(
                text="t2",
                segment_hash="bbbb",
                segments_dir=segments,
                metadata={"script_idx": 1, "chunk_idx": 0, "script_path": "x"},
            ),
        ]

        assert not timing_manifest_path(wav_path).exists()

        engine = MagicMock()
        engine.config.batch_size = 2
        process_audio_pipeline(engine, [(wav_path, tasks)])

        # manifest now exists without any synthesis being invoked.
        assert timing_manifest_path(wav_path).exists()
        engine.synthesize.assert_not_called()
        engine.clone_voice.assert_not_called()


class TestContentFingerprint:
    """regenerated segment wavs must invalidate the chapter wav."""

    def test_regenerated_chunk_is_detected_as_stale(self, tmp_path):
        from autiobook.pooling import _chapter_fingerprint

        perform = tmp_path / "perform"
        segments = perform / "segments"
        segments.mkdir(parents=True)
        _write_wav(segments / "aaaa.wav", 1.0)
        _write_wav(segments / "bbbb.wav", 1.0)

        before = _chapter_fingerprint(segments, ["aaaa", "bbbb"])

        # simulate regenerating a chunk wav with different audio content.
        _write_wav(segments / "aaaa.wav", 1.0)
        # touch alone would not change bytes; write different samples.
        import numpy as np
        import soundfile as sf  # type: ignore

        sf.write(
            str(segments / "aaaa.wav"),
            np.ones(24000, dtype=np.float32) * 0.5,
            24000,
        )

        after = _chapter_fingerprint(segments, ["aaaa", "bbbb"])
        assert before != after

    def test_fingerprint_stable_across_calls(self, tmp_path):
        from autiobook.pooling import _chapter_fingerprint

        segments = tmp_path / "segments"
        segments.mkdir()
        _write_wav(segments / "aaaa.wav", 1.0)
        h1 = _chapter_fingerprint(segments, ["aaaa"])
        h2 = _chapter_fingerprint(segments, ["aaaa"])
        assert h1 == h2

    def test_fingerprint_falls_back_when_segment_missing(self, tmp_path):
        """first-time build path: no segments on disk yet."""
        from autiobook.pooling import _chapter_fingerprint
        from autiobook.resume import compute_hash

        segments = tmp_path / "segments"
        segments.mkdir()
        # no wav written
        fp = _chapter_fingerprint(segments, ["aaaa", "bbbb"])
        assert fp == compute_hash(["aaaa", "bbbb"])
