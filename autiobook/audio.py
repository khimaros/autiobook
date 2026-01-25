"""audio processing utilities."""

from pathlib import Path

import numpy as np
import soundfile as sf

from .config import (
    SAMPLE_RATE,
    SEGMENTS_DIR,
    WAV_EXT,
)


def concatenate_audio(
    audio_chunks: list[np.ndarray],
    sample_rate: int = SAMPLE_RATE,
    pause_duration_ms: int = 500,
) -> np.ndarray:
    """concatenate audio chunks with pauses between them."""
    if not audio_chunks:
        return np.array([], dtype=np.float32)

    pause_samples = int(sample_rate * pause_duration_ms / 1000)
    pause = np.zeros(pause_samples, dtype=np.float32)

    result = []
    for i, chunk in enumerate(audio_chunks):
        result.append(chunk)
        if i < len(audio_chunks) - 1:
            result.append(pause)

    return np.concatenate(result)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """normalize audio levels to prevent clipping."""
    if audio.size == 0:
        return audio

    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95

    return audio


def get_segments_dir(workdir: Path) -> Path:
    """get the central segments cache directory."""
    return workdir / SEGMENTS_DIR


def get_segment_path(segments_dir: Path, segment_hash: str) -> Path:
    """get the path for a segment file."""
    return segments_dir / f"{segment_hash}{WAV_EXT}"


def check_segment_exists(segments_dir: Path, segment_hash: str) -> bool:
    """check if a segment exists in the cache."""
    return get_segment_path(segments_dir, segment_hash).exists()


def save_segment(
    segments_dir: Path, segment_hash: str, audio: np.ndarray, sample_rate: int
) -> None:
    """save a segment to the central cache."""
    segments_dir.mkdir(parents=True, exist_ok=True)
    path = get_segment_path(segments_dir, segment_hash)
    sf.write(str(path), audio, sample_rate)


def load_segment(segments_dir: Path, segment_hash: str) -> np.ndarray:
    """load a segment from the central cache."""
    path = get_segment_path(segments_dir, segment_hash)
    if not path.exists():
        raise FileNotFoundError(f"segment {segment_hash} not found in {segments_dir}")
    audio, _ = sf.read(str(path))
    return audio.astype(np.float32)
