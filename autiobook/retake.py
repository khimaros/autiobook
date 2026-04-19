"""detect and remove corrupted segment wavs so they can be regenerated.

heuristics are calibrated against 1207 walkaway segments; thresholds err on the
side of false negatives so we don't delete good takes."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf  # type: ignore

from .config import REJECTED_DIR, SAMPLE_RATE, SEGMENTS_DIR, WAV_EXT
from .resume import get_command_dir

# thresholds
SILENCE_MEAN_ABS = 0.001
CLICK_FIRST_SAMPLE = 0.05
TRUNC_LAST_SAMPLE = 0.05
CLIP_COUNT = 10
CLIP_THRESHOLD = 0.99

# "noisy" detects flat, loud-throughout audio — characteristic of voice-clone
# ref-text leaks. tight rule fires on any duration; loose rule requires ≥3s
# because short clips don't give the statistics room to settle.
NOISY_CREST_TIGHT = 3.5
NOISY_MED_TIGHT = 0.08
NOISY_CREST_LOOSE = 4.5
NOISY_MED_LOOSE = 0.07
NOISY_LOOSE_MIN_DUR_S = 3.0


@dataclass
class SegmentMetrics:
    path: Path
    duration_s: float
    mean_abs: float
    peak: float
    first_sample: float
    last_sample: float
    n_clipped: int
    categories: list[str] = field(default_factory=list)


def categorize_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[str]:
    """classify an in-memory audio array against each artifact heuristic."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    if n == 0:
        return ["silent"]
    fs = float(audio[0])
    ls = float(audio[-1])
    mean_abs = float(np.mean(np.abs(audio)))
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio * audio)))
    crest = peak / rms if rms > 0 else float("inf")
    median_abs = float(np.median(np.abs(audio)))
    duration_s = n / sr if sr else 0.0
    n_clipped = int(np.sum(np.abs(audio) >= CLIP_THRESHOLD))

    cats: list[str] = []
    if mean_abs < SILENCE_MEAN_ABS:
        cats.append("silent")
    if abs(fs) > CLICK_FIRST_SAMPLE:
        cats.append("click")
    if abs(ls) > TRUNC_LAST_SAMPLE:
        cats.append("truncated")
    if n_clipped > CLIP_COUNT:
        cats.append("clipping")
    tight = crest < NOISY_CREST_TIGHT and median_abs > NOISY_MED_TIGHT
    loose = (
        crest < NOISY_CREST_LOOSE
        and median_abs > NOISY_MED_LOOSE
        and duration_s >= NOISY_LOOSE_MIN_DUR_S
    )
    if tight or loose:
        cats.append("noisy")
    return cats


def _safe_label(label: str) -> str:
    """slug a label for safe use in filenames."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:64]


def save_reject(
    reject_dir: Path,
    audio: np.ndarray,
    sr: int,
    categories: list[str],
    label: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """archive a failed take (wav + json sidecar) so it can be audited later.

    filename format: <ms>_<label>_<cats>.wav — monotonic ms keeps retries of
    the same label distinct and sortable.
    """
    reject_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{time.monotonic_ns()}_{_safe_label(label)}_{'-'.join(categories)}"
    wav_path = reject_dir / f"{stem}{WAV_EXT}"
    sf.write(str(wav_path), audio, sr)

    sidecar = {
        "label": label,
        "categories": categories,
        "sample_rate": sr,
        "duration_s": float(len(audio) / sr) if sr else 0.0,
        "timestamp": time.time(),
        "metadata": metadata or {},
    }
    (reject_dir / f"{stem}.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False)
    )
    return wav_path


def get_reject_dir(workdir: Path, phase: str) -> Path:
    """reject dir for a given phase (emote, perform, synthesize)."""
    return get_command_dir(workdir, phase) / REJECTED_DIR


def analyze_segment(path: Path) -> SegmentMetrics:
    """compute quality metrics and categorize artifacts for a single wav."""
    audio, sr = sf.read(str(path), dtype="float32")
    audio = cast(np.ndarray, audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    n = len(audio)
    fs = float(audio[0]) if n else 0.0
    ls = float(audio[-1]) if n else 0.0
    mean_abs = float(np.mean(np.abs(audio))) if n else 0.0
    peak = float(np.max(np.abs(audio))) if n else 0.0
    n_clipped = int(np.sum(np.abs(audio) >= CLIP_THRESHOLD)) if n else 0

    return SegmentMetrics(
        path=path,
        duration_s=float(n / sr) if sr else 0.0,
        mean_abs=mean_abs,
        peak=peak,
        first_sample=fs,
        last_sample=ls,
        n_clipped=n_clipped,
        categories=categorize_audio(audio, sr),
    )


def find_offenders(segments_dir: Path) -> list[SegmentMetrics]:
    """scan all wavs and return those flagged by any heuristic."""
    results = []
    for p in sorted(segments_dir.glob(f"*{WAV_EXT}")):
        m = analyze_segment(p)
        if m.categories:
            results.append(m)
    return results


def _build_hash_index(command_dir: Path) -> dict[str, tuple[str, int]]:
    """map segment hash -> (script_path, script_idx) by scanning timing manifests."""
    mapping: dict[str, tuple[str, int]] = {}
    for p in command_dir.glob("*.wav.timing.json"):
        try:
            manifest = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for c in manifest.get("chunks", []):
            h = c.get("hash")
            sp = c.get("script_path")
            si = c.get("script_idx")
            if h and sp and si is not None and h not in mapping:
                mapping[h] = (sp, int(si))
    return mapping


def _load_script_segment(script_path: str, script_idx: int) -> dict[str, str] | None:
    """fetch a script segment (text, speaker, instruction), or None if unavailable."""
    p = Path(script_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    segs = data.get("segments", [])
    if 0 <= script_idx < len(segs):
        s = segs[script_idx]
        return {
            "text": str(s.get("text") or ""),
            "speaker": str(s.get("speaker") or ""),
            "instruction": str(s.get("instruction") or ""),
        }
    return None


def format_metrics(m: SegmentMetrics, segment: dict[str, str] | None = None) -> str:
    cats = ",".join(m.categories)
    short = m.path.stem[:16]
    line = (
        f"[{cats:28s}] {short} "
        f"ma={m.mean_abs:.4f} fs={m.first_sample:+.3f} "
        f"ls={m.last_sample:+.3f} clip={m.n_clipped} dur={m.duration_s:.1f}s"
    )
    if segment:
        speaker = segment.get("speaker") or "?"
        instruction = segment.get("instruction") or ""
        attribution = f"[{speaker}]"
        if instruction:
            attribution += f" ({instruction})"
        line += f"\n    {attribution}"
        text = segment.get("text") or ""
        if text:
            snippet = text.strip().replace("\n", " ")
            if len(snippet) > 100:
                snippet = snippet[:97] + "..."
            line += f"\n    text: {snippet}"
    return line


def _regenerate(
    workdir: Path,
    command: str,
    chapters: list[int] | None,
    config: object,
    verbose: bool,
    only_hashes: set[str],
) -> None:
    """re-run synthesis, scoped to the specific hashes that were deleted."""
    if command == "perform":
        from .dramatize import run_performance

        run_performance(
            workdir,
            chapters=chapters,
            config=config,
            verbose=verbose,
            retake=True,
            only_hashes=only_hashes,
        )
    elif command == "synthesize":
        from .tts import synthesize_chapters

        synthesize_chapters(
            workdir,
            config,
            chapters,
            retake=True,
            only_hashes=only_hashes,
        )
    else:
        print(f"retake: regeneration not supported for command '{command}'")


def run_retake(
    workdir: Path,
    command: str = "perform",
    chapters: list[int] | None = None,
    config: object = None,
    verbose: bool = False,
    prune: bool = False,
    dry_run: bool = False,
) -> None:
    """scan segments, delete offenders, and regenerate only the deleted ones."""
    command_dir = get_command_dir(workdir, command)
    segments_dir = command_dir / SEGMENTS_DIR

    if not segments_dir.exists():
        print(f"retake: no segments directory at {segments_dir}")
        return

    offenders = find_offenders(segments_dir)
    if not offenders:
        print("retake: no offenders found")
        return

    index = _build_hash_index(command_dir)

    counts: dict[str, int] = {}
    for o in offenders:
        for c in o.categories:
            counts[c] = counts.get(c, 0) + 1

    print(f"retake: {len(offenders)} offender(s) in {segments_dir}")
    for cat in sorted(counts):
        print(f"  {cat}: {counts[cat]}")
    print()

    for o in offenders:
        segment = None
        sm = index.get(o.path.stem)
        if sm:
            segment = _load_script_segment(*sm)
        print(format_metrics(o, segment))

    if dry_run:
        print(
            f"\nretake: dry run - {len(offenders)} wav(s) would be deleted and regenerated"
        )
        return

    reject_dir = get_reject_dir(workdir, command)
    deleted_hashes: set[str] = set()
    for o in offenders:
        audio, sr = sf.read(str(o.path), dtype="float32")
        audio = cast(np.ndarray, audio)
        segment = None
        sm = index.get(o.path.stem)
        if sm:
            segment = _load_script_segment(*sm)
        save_reject(
            reject_dir,
            audio,
            sr,
            o.categories,
            o.path.stem,
            metadata={
                "phase": command,
                "hash": o.path.stem,
                "duration_s": o.duration_s,
                "mean_abs": o.mean_abs,
                "peak": o.peak,
                "first_sample": o.first_sample,
                "last_sample": o.last_sample,
                "n_clipped": o.n_clipped,
                "script_path": sm[0] if sm else None,
                "script_idx": sm[1] if sm else None,
                "segment": segment,
            },
        )
        o.path.unlink()
        deleted_hashes.add(o.path.stem)

    if prune:
        print(f"\nretake: deleted {len(offenders)} wav(s); skipping regeneration")
        return

    print(f"\nretake: deleted {len(offenders)} wav(s); regenerating...")
    _regenerate(workdir, command, chapters, config, verbose, deleted_hashes)


def cmd_retake(args):
    """identify corrupted segment wavs; delete and regenerate unless --dry-run."""
    from .utils import get_chapters, get_clone_config, get_tts_config

    command = args.command
    config = get_clone_config(args) if command == "perform" else get_tts_config(args)
    run_retake(
        workdir=Path(args.workdir),
        command=command,
        chapters=get_chapters(args),
        config=config,
        verbose=getattr(args, "verbose", False),
        prune=getattr(args, "prune", False),
        dry_run=args.dry_run,
    )
