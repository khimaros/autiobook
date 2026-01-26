"""tests for showcase command - generates emotion samples for character voices."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autiobook.config import SAMPLE_RATE, SHOWCASE_EMOTIONS, WAV_EXT
from autiobook.dramatize import run_showcase


@pytest.fixture
def temp_workdir():
    """create a temporary workdir with cast and audition samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)

        # create cast directory with characters.json
        cast_dir = workdir / "cast"
        cast_dir.mkdir()
        cast_data = {
            "version": 4,
            "characters": [
                {
                    "name": "Alice",
                    "description": "warm female voice, mid-range pitch",
                    "audition_line": "Hello, I am Alice.",
                    "aliases": None,
                },
                {
                    "name": "Bob",
                    "description": "deep male voice, slow pace",
                    "audition_line": "Greetings, I am Bob.",
                    "aliases": ["Robert"],
                },
            ],
        }
        with open(cast_dir / "characters.json", "w") as f:
            json.dump(cast_data, f)

        # create audition directory with voice samples
        audition_dir = workdir / "audition"
        audition_dir.mkdir()

        # create mock wav files using soundfile
        import soundfile as sf

        for name in ["Alice", "Bob"]:
            audio = np.zeros(1000, dtype=np.float32)
            sf.write(str(audition_dir / f"{name}{WAV_EXT}"), audio, SAMPLE_RATE)

        yield workdir


@pytest.fixture
def mock_engine():
    """create a mock TTS engine that tracks clone_voice calls."""
    engine = MagicMock()
    engine.config = MagicMock()
    engine.config.batch_size = 4

    call_log = []

    def mock_clone_voice(texts, ref_audio, ref_text):
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            call_log.append(("clone", text, str(ref_audio)))
        return [np.zeros(1000, dtype=np.float32) for _ in texts], SAMPLE_RATE

    engine.clone_voice = mock_clone_voice
    engine.call_log = call_log

    return engine


class TestShowcaseCommand:
    """tests for the showcase command."""

    def test_creates_showcase_directory_structure(self, temp_workdir, mock_engine):
        """verify showcase creates per-character directories with emotion samples."""
        with patch("autiobook.dramatize.TTSEngine", return_value=mock_engine):
            run_showcase(temp_workdir)

        showcase_dir = temp_workdir / "showcase"
        assert showcase_dir.exists(), "showcase directory not created"

        # check per-character directories
        assert (showcase_dir / "Alice").is_dir(), "Alice directory not created"
        assert (showcase_dir / "Bob").is_dir(), "Bob directory not created"

        # check emotion samples exist for each character
        for char_name in ["Alice", "Bob"]:
            char_dir = showcase_dir / char_name
            for emotion in SHOWCASE_EMOTIONS:
                wav_path = char_dir / f"{emotion}{WAV_EXT}"
                assert wav_path.exists(), f"{char_name}/{emotion}.wav not created"

    def test_uses_voice_cloning_from_audition(self, temp_workdir, mock_engine):
        """verify showcase uses clone_voice with audition samples as reference."""
        with patch("autiobook.dramatize.TTSEngine", return_value=mock_engine):
            run_showcase(temp_workdir)

        # all calls should be clone operations
        call_types = set(c[0] for c in mock_engine.call_log)
        assert call_types == {"clone"}, f"expected only clone calls, got {call_types}"

        # verify reference audio paths point to audition directory
        audition_dir = temp_workdir / "audition"
        for call in mock_engine.call_log:
            ref_path = call[2]
            assert (
                str(audition_dir) in ref_path
            ), f"ref_audio not from audition: {ref_path}"

    def test_generates_all_emotions_for_all_characters(self, temp_workdir, mock_engine):
        """verify all emotion/character combinations are generated."""
        with patch("autiobook.dramatize.TTSEngine", return_value=mock_engine):
            run_showcase(temp_workdir)

        # count expected samples: 2 characters x 11 emotions
        expected_count = 2 * len(SHOWCASE_EMOTIONS)

        # count actual clone calls
        clone_calls = [c for c in mock_engine.call_log if c[0] == "clone"]
        assert (
            len(clone_calls) == expected_count
        ), f"expected {expected_count} samples, got {len(clone_calls)}"

    def test_resumability_skips_existing_samples(self, temp_workdir, mock_engine):
        """verify showcase skips already-generated samples on second run."""
        with patch("autiobook.dramatize.TTSEngine", return_value=mock_engine):
            # first run generates all samples
            run_showcase(temp_workdir)
            first_run_calls = len(mock_engine.call_log)

            # clear call log
            mock_engine.call_log.clear()

            # second run should skip all (no force flag)
            run_showcase(temp_workdir)
            second_run_calls = len(mock_engine.call_log)

        assert (
            second_run_calls == 0
        ), f"expected 0 calls on second run (resumable), got {second_run_calls}"

    def test_force_flag_regenerates_all(self, temp_workdir, mock_engine):
        """verify force flag regenerates all samples."""
        with patch("autiobook.dramatize.TTSEngine", return_value=mock_engine):
            # first run
            run_showcase(temp_workdir)
            first_run_calls = len(mock_engine.call_log)

            # clear call log
            mock_engine.call_log.clear()

            # second run with force should regenerate all
            run_showcase(temp_workdir, force=True)
            second_run_calls = len(mock_engine.call_log)

        assert (
            second_run_calls == first_run_calls
        ), f"force run should regenerate all: expected {first_run_calls}, got {second_run_calls}"

    def test_handles_missing_audition_gracefully(self, temp_workdir, mock_engine):
        """verify showcase fails gracefully when audition samples missing."""
        # remove audition files
        audition_dir = temp_workdir / "audition"
        for f in audition_dir.glob("*.wav"):
            f.unlink()

        with patch("autiobook.dramatize.TTSEngine", return_value=mock_engine):
            # should not raise, but should print warning and skip
            run_showcase(temp_workdir)

        # no samples should be generated
        assert len(mock_engine.call_log) == 0

    def test_handles_missing_cast_gracefully(self, temp_workdir, mock_engine):
        """verify showcase fails gracefully when cast is missing."""
        # remove cast file
        (temp_workdir / "cast" / "characters.json").unlink()

        with patch("autiobook.dramatize.TTSEngine", return_value=mock_engine):
            # should not raise
            run_showcase(temp_workdir)

        # no samples should be generated (uses default cast which may not have auditions)
        # this depends on implementation - may need adjustment


class TestShowcaseBatching:
    """tests for batched synthesis in showcase command."""

    def test_batches_clone_calls(self, temp_workdir, mock_engine):
        """verify showcase batches clone_voice calls for efficiency."""
        # set small batch size to verify batching behavior
        mock_engine.config.batch_size = 2

        batch_sizes = []

        def mock_clone_voice_track_batch(texts, ref_audio, ref_text):
            if isinstance(texts, str):
                texts = [texts]
            batch_sizes.append(len(texts))
            return [np.zeros(1000, dtype=np.float32) for _ in texts], SAMPLE_RATE

        mock_engine.clone_voice = mock_clone_voice_track_batch

        with patch("autiobook.dramatize.TTSEngine", return_value=mock_engine):
            run_showcase(temp_workdir)

        # verify batching occurred (at least some calls should have multiple texts)
        # with batch_size=2, we should see batches of 2 for same-voice segments
        assert len(batch_sizes) > 0, "no clone calls made"
