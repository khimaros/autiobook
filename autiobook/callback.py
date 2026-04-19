"""audition/emote quality checks — the casting equivalent of retake.

mirrors retake.py: same heuristics (categorize_audio), same --dry-run/--prune
surface, but scans voice-sample wavs in audition/ (base files) and emote/
(per-emotion variants) rather than chapter segment caches. invoked inline
during generation (via --callback) or as a post-hoc scan via the `callback`
subcommand.
"""

from pathlib import Path
from typing import Any

from .config import WAV_EXT
from .resume import get_command_dir
from .retake import (
    SegmentMetrics,
    analyze_segment,
    categorize_audio,
    save_reject,
)

# inline guard: retry bad generations before they reach disk
CALLBACK_MAX_ATTEMPTS = 5


def generate_with_callback(
    gen_fn,
    engine,
    label: str = "",
    max_attempts: int = CALLBACK_MAX_ATTEMPTS,
    verbose: bool = False,
    reject_dir: Path | None = None,
    metadata: dict[str, Any] | None = None,
):
    """run gen_fn() until categorize_audio returns no categories.

    bumps engine.config.seed between attempts when a seed is set; raises
    RuntimeError if all attempts are still flagged as corrupted. when
    reject_dir is provided, each rejected take is archived alongside a json
    sidecar (seed + metadata) for later forensic review.
    """
    cats: list[str] = []
    for attempt in range(1, max_attempts + 1):
        audio, sr = gen_fn()
        cats = categorize_audio(audio, sr)
        if not cats:
            if verbose and attempt > 1:
                print(f"  callback: {label} passed on attempt {attempt}/{max_attempts}")
            return audio, sr
        print(
            f"  callback: {label} rejected ({','.join(cats)}) "
            f"attempt {attempt}/{max_attempts}"
        )
        if reject_dir is not None:
            save_reject(
                reject_dir,
                audio,
                sr,
                cats,
                label,
                metadata={
                    **(metadata or {}),
                    "seed": int(getattr(engine.config, "seed", 0) or 0),
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                },
            )
        if getattr(engine.config, "seed", 0) > 0:
            engine.config.seed += 1
    raise RuntimeError(
        f"callback: {label} failed audio quality after {max_attempts} attempts "
        f"(last: {','.join(cats)})"
    )


def _scan_dir(root: Path) -> list[SegmentMetrics]:
    """scan a voices dir, return offenders."""
    if not root.exists():
        return []
    out = []
    for p in sorted(root.glob(f"*{WAV_EXT}")):
        m = analyze_segment(p)
        if m.categories:
            out.append(m)
    return out


def find_offenders(workdir: Path) -> dict[str, list[SegmentMetrics]]:
    """scan audition/ and emote/ wavs. returns {phase: [offenders]}."""
    return {
        "audition": _scan_dir(get_command_dir(workdir, "audition")),
        "emote": _scan_dir(get_command_dir(workdir, "emote")),
    }


def run_callback(
    workdir: Path,
    design_config=None,
    verbose: bool = False,
    prune: bool = False,
    dry_run: bool = False,
) -> None:
    """scan audition/emote wavs, delete offenders, and regenerate via generators.

    mirrors retake: dry-run reports only, prune deletes but skips regen.
    """
    from .retake import format_metrics

    offenders = find_offenders(workdir)
    total = sum(len(v) for v in offenders.values())
    if total == 0:
        print("callback: no offenders found")
        return

    for phase, items in offenders.items():
        if not items:
            continue
        counts: dict[str, int] = {}
        for o in items:
            for c in o.categories:
                counts[c] = counts.get(c, 0) + 1
        print(f"callback: {len(items)} offender(s) in {phase}/")
        for cat in sorted(counts):
            print(f"  {cat}: {counts[cat]}")
        for o in items:
            print(format_metrics(o))
        print()

    if dry_run:
        print(f"callback: dry run - {total} wav(s) would be deleted and regenerated")
        return

    for items in offenders.values():
        for o in items:
            o.path.unlink()

    if prune:
        print(f"callback: deleted {total} wav(s); skipping regeneration")
        return

    print(f"callback: deleted {total} wav(s); regenerating...")

    # audition base files live in audition/; emote variants in emote/.
    # deleting them means the run_* generators see resume-state fresh but file
    # missing, and regen.
    if offenders["audition"]:
        from .audition import run_audition

        run_audition(workdir, verbose=verbose, config=design_config, callback=True)

    if offenders["emote"]:
        from .dramatize import run_emotes

        run_emotes(workdir, verbose=verbose, config=design_config, callback=True)


def cmd_callback(args):
    """post-hoc quality scan for audition/emote samples."""
    from .utils import get_design_config

    run_callback(
        workdir=Path(args.workdir),
        design_config=get_design_config(args),
        verbose=getattr(args, "verbose", False),
        prune=getattr(args, "prune", False),
        dry_run=args.dry_run,
    )
