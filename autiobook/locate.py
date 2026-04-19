"""map an audio time position back to the originating script segment."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .audio import get_segment_path
from .config import SEGMENTS_DIR, WAV_EXT
from .pooling import timing_manifest_path


@dataclass
class SegmentLocation:
    """a chunk located at a given time inside a chapter wav.

    chunk_wav is the most actionable field: the segment wav file to inspect."""

    chunk_wav: Path
    chunk_hash: str
    chunk_start_s: float
    chunk_end_s: float
    script_idx: Optional[int]
    chunk_idx: Optional[int]
    speaker: Optional[str]
    text: Optional[str]
    instruction: Optional[str]


_TIME_RE = re.compile(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_time(s: str) -> float:
    """accept '1:23.5', '83.5', or '83' and return seconds."""
    m = _TIME_RE.match(s.strip())
    if not m:
        raise ValueError(f"invalid time '{s}'; use seconds or m:ss")
    minutes, seconds = m.group(1), m.group(2)
    total = float(seconds)
    if minutes:
        total += int(minutes) * 60
    return total


def _resolve_segments_dir(wav_path: Path) -> Path:
    """find the shared segments cache for a chapter wav.

    segments/ sits next to the chapter wav (inside perform/, synthesize/, etc.)."""
    return wav_path.parent / SEGMENTS_DIR


def _load_script_segment(
    script_path: Path, script_idx: int
) -> tuple[str, str, str] | None:
    """return (speaker, text, instruction) for a script segment, if available."""
    if not script_path.exists():
        return None
    data = json.loads(script_path.read_text())
    segments = data.get("segments", [])
    if not (0 <= script_idx < len(segments)):
        return None
    s = segments[script_idx]
    return s.get("speaker", ""), s.get("text", ""), s.get("instruction", "")


def locate_segment(wav_path: Path, time_s: float) -> SegmentLocation:
    """look up the chunk (and script segment) at `time_s` within `wav_path`."""
    manifest_path = timing_manifest_path(wav_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no timing manifest at {manifest_path}; re-run perform to regenerate"
        )
    manifest = json.loads(manifest_path.read_text())
    chunks: list[dict[str, Any]] = manifest["chunks"]
    if not chunks:
        raise ValueError(f"empty manifest: {manifest_path}")

    total = chunks[-1]["end_s"]
    if time_s < 0 or time_s > total:
        raise ValueError(f"time {time_s:.3f}s out of range [0, {total:.3f}]")

    # linear scan is fine (~1000 chunks per chapter); binary search if it grows.
    chosen = chunks[-1]
    for c in chunks:
        if c["end_s"] >= time_s:
            chosen = c
            break

    segments_dir = _resolve_segments_dir(wav_path)
    chunk_wav = get_segment_path(segments_dir, chosen["hash"])

    speaker = text = instruction = None
    script_idx = chosen.get("script_idx")
    script_path_str = chosen.get("script_path")
    if script_idx is not None and script_path_str:
        found = _load_script_segment(Path(script_path_str), script_idx)
        if found:
            speaker, text, instruction = found

    return SegmentLocation(
        chunk_wav=chunk_wav,
        chunk_hash=chosen["hash"],
        chunk_start_s=chosen["start_s"],
        chunk_end_s=chosen["end_s"],
        script_idx=script_idx,
        chunk_idx=chosen.get("chunk_idx"),
        speaker=speaker,
        text=text,
        instruction=instruction,
    )


def format_location(loc: SegmentLocation) -> str:
    """human-readable rendering for the CLI."""
    lines = [str(loc.chunk_wav)]
    lines.append(
        f"  time:    {loc.chunk_start_s:.3f}s - {loc.chunk_end_s:.3f}s"
        f" (duration {loc.chunk_end_s - loc.chunk_start_s:.3f}s)"
    )
    lines.append(f"  hash:    {loc.chunk_hash}")
    if loc.script_idx is not None:
        lines.append(f"  segment: #{loc.script_idx} chunk {loc.chunk_idx}")
    if loc.speaker:
        lines.append(f"  speaker: {loc.speaker}")
    if loc.instruction:
        lines.append(f"  emotion: {loc.instruction}")
    if loc.text:
        lines.append(f"  text:    {loc.text}")
    return "\n".join(lines)


# re-export for consumers that only need the wav extension convention
__all__ = [
    "SegmentLocation",
    "locate_segment",
    "parse_time",
    "format_location",
    "WAV_EXT",
]
