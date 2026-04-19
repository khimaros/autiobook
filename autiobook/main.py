"""cli entry point for autiobook."""

import argparse
import os
import sys
from pathlib import Path

# load .env before other imports so env vars are available for config
from .env import load_env

load_env()

from .config import (  # noqa: E402
    BASE_MODEL,
    DEFAULT_BITRATE,
    DEFAULT_MODEL,
    VOICE_DESIGN_MODEL,
)
from .epub import ensure_extracted, parse_epub  # noqa: E402
from .export import export_audiobook  # noqa: E402
from .utils import (  # noqa: E402
    Logger,
    add_common_args,
    find_redo_phase,
    get_chapters,
    get_pipeline_paths,
    get_tts_config,
)


def cmd_download(args):
    """download tts model weights."""
    from huggingface_hub import snapshot_download

    models = []
    if args.all:
        models = [DEFAULT_MODEL, VOICE_DESIGN_MODEL, BASE_MODEL]
    else:
        models = [args.model]

    for model in models:
        print(f"download: downloading model {model}...")
        path = snapshot_download(repo_id=model)
        print(f"download: model downloaded to {path}")


def cmd_chapters(args):
    """list chapters in an epub file."""
    epub_path = Path(args.epub)
    book, cover_data = parse_epub(epub_path)

    print(f"chapters: title: {book.title}")
    print(f"chapters: author: {book.author}")
    print(f"chapters: language: {book.language}")
    print(f"chapters: count: {len(book.chapters)}")
    print(f"chapters: cover: {'yes' if cover_data else 'no'}")
    print()

    for chapter in book.chapters:
        print(f"  {chapter.index:2d}. {chapter.title} ({chapter.word_count} words)")


def cmd_extract(args):
    """extract chapter text from epub to workdir."""
    epub_path = Path(args.epub)
    workdir = Path(args.output)
    ensure_extracted(epub_path, workdir, force=args.force)
    print("extract: done")


class StepComplete(Exception):
    """raised to exit after a pipeline phase when --step is active."""

    def __init__(self, phase: str):
        self.phase = phase
        super().__init__(f"step complete: {phase}")


def _run_pipeline(args, process_fn, name, phases=None):
    """common helper for full pipelines."""
    epub_path, workdir = get_pipeline_paths(args)
    export_dir = workdir / "export"
    chapters = get_chapters(args)
    step = getattr(args, "step", False)
    redo = getattr(args, "redo", False)

    from .utils import dir_mtime

    # --redo: find last completed phase and force it
    force = args.force
    redo_phase = None
    if redo and phases:
        redo_phase = find_redo_phase(workdir, phases)
        if redo_phase:
            print(f"redo: re-running phase '{redo_phase}'")
        else:
            print("redo: no completed phases found, running from start")

    force_extract = force or redo_phase == "extract"
    extract_before = dir_mtime(workdir / "extract")
    ensure_extracted(epub_path, workdir, force=force_extract)
    if step and dir_mtime(workdir / "extract") > extract_before:
        print("step: extract complete. re-run to continue.")
        return

    Logger.init(workdir)

    config = get_tts_config(args)

    before = dir_mtime(workdir)
    try:
        process_fn(workdir, config, chapters, redo_phase=redo_phase)
    except StepComplete as e:
        print(f"step: {e.phase} complete. re-run to continue.")
        return

    if step and dir_mtime(workdir) > before:
        print("step: processing complete. re-run to export.")
        return

    force_export = force or redo_phase == "export"
    print(f"export: exporting chapters to {export_dir}/...")
    new, skipped = export_audiobook(
        workdir, export_dir, args.bitrate, force=force_export, m4b=args.m4b
    )

    msg = f"{name}: done - {new} chapter(s) exported"
    if skipped > 0:
        msg += f" ({skipped} skipped)"
    print(msg)


def cmd_dramatize(args):
    """generate script and cast using LLM."""
    from .dramatize import dramatize_book
    from .utils import get_clone_config, get_design_config

    design_config = get_design_config(args)
    clone_config = get_clone_config(args)

    step = getattr(args, "step", False)
    emotions = getattr(args, "emotions", False)
    preset_voices = getattr(args, "preset_voices", False)
    directed = getattr(args, "directed", False)
    phases = ["extract", "cast", "audition"]
    if emotions:
        phases.append("emote")
    phases.extend(["script", "revise", "perform", "retake", "export"])

    # --strict rolls up all validation checks
    strict = getattr(args, "strict", False)
    revise = args.revise or strict
    retake = args.retake or strict
    callback = getattr(args, "callback", False) or strict

    def process_fn(workdir, config, chapters, redo_phase=None):
        print(f"dramatize: dramatizing chapters in {workdir}...")
        dramatize_book(
            workdir,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            chapters=chapters,
            design_config=design_config,
            clone_config=clone_config,
            pooled=args.pooled,
            verbose=args.verbose,
            force=args.force,
            thinking_budget=args.thinking_budget,
            revise=revise,
            step=step,
            redo_phase=redo_phase,
            retake=retake,
            callback=callback,
            emotions=emotions,
            preset_voices=preset_voices,
            directed=directed,
        )

    _run_pipeline(args, process_fn, "dramatize", phases=phases)


def cmd_convert(args):
    """run full conversion pipeline."""
    from .tts import synthesize_chapters

    phases = ["extract", "synthesize", "retake", "export"]

    strict = getattr(args, "strict", False)
    retake = args.retake or strict

    def process_fn(workdir, config, chapters, redo_phase=None):
        force = args.force or redo_phase == "synthesize"
        print(f"synthesize: synthesizing chapters in {workdir}/synthesize/...")
        synthesize_chapters(
            workdir,
            config,
            chapters,
            args.instruct,
            args.pooled,
            force=force,
            retake=retake,
        )

        from .retake import run_retake

        run_retake(
            workdir,
            command="synthesize",
            chapters=chapters,
            config=config,
            verbose=args.verbose,
        )

    _run_pipeline(args, process_fn, "convert", phases=phases)


def cmd_synthesize(args):
    """convert text files to wav audio."""
    from .tts import synthesize_chapters

    workdir = Path(args.workdir)
    chapters = get_chapters(args)
    config = get_tts_config(args)

    print(f"synthesize: synthesizing chapters in {workdir}/synthesize/...")
    synthesize_chapters(
        workdir,
        config,
        chapters,
        args.instruct,
        args.pooled,
        force=args.force,
        retake=getattr(args, "retake", False),
    )

    print("synthesize: done")


def cmd_export(args):
    """convert wav files to mp3 with metadata."""
    workdir = Path(args.workdir)
    output_dir = Path(args.output) if args.output else workdir / "export"

    print(f"export: exporting chapters to {output_dir}/...")
    new, skipped = export_audiobook(
        workdir, output_dir, args.bitrate, force=args.force, m4b=args.m4b
    )

    msg = f"export: {new} chapter(s) exported"
    if skipped > 0:
        msg += f" ({skipped} skipped)"
    print(msg)


def cmd_clean(args):
    """remove intermediate files (segment caches)."""
    import shutil

    from .resume import get_command_dir

    workdir = Path(args.workdir)
    clean_dirs = [
        get_command_dir(workdir, "perform") / "segments",
        get_command_dir(workdir, "synthesize") / "segments",
    ]

    to_remove = [d for d in clean_dirs if d.exists() and d.is_dir()]

    if not to_remove:
        print("clean: no segment caches found")
        return

    for d in to_remove:
        if args.dry_run:
            print(f"clean: would remove {d}")
        else:
            shutil.rmtree(d)
            print(f"clean: removed {d}")


def cmd_locate(args):
    """look up the segment wav at an audio time position."""
    from .locate import format_location, locate_segment, parse_time

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"locate: wav not found: {wav_path}", file=sys.stderr)
        sys.exit(1)

    try:
        t = parse_time(args.time)
        loc = locate_segment(wav_path, t)
    except (FileNotFoundError, ValueError) as e:
        print(f"locate: {e}", file=sys.stderr)
        sys.exit(1)

    print(format_location(loc))


def _cmd_retake(args):
    from .retake import cmd_retake

    cmd_retake(args)


def _cmd_callback(args):
    from .callback import cmd_callback

    cmd_callback(args)


def _cmd_design(args):
    from .design import cmd_design

    cmd_design(args)


def _cmd_cast(args):
    from .dramatize import cmd_cast

    cmd_cast(args)


def _cmd_audition(args):
    from .audition import cmd_audition

    cmd_audition(args)


def _cmd_emote(args):
    from .dramatize import cmd_emote

    cmd_emote(args)


def _cmd_script(args):
    from .dramatize import cmd_script

    cmd_script(args)


def _cmd_perform(args):
    from .dramatize import cmd_perform

    cmd_perform(args)


def _cmd_revise(args):
    from .dramatize import cmd_revise

    cmd_revise(args)


def main():
    parser = argparse.ArgumentParser(
        prog="autiobook",
        description="convert epub files to audiobooks using qwen3-tts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = {
        "download": (
            cmd_download,
            "download tts model weights",
            [("runtime",)],
            [
                (("-m", "--model"), {"default": DEFAULT_MODEL, "help": "model name"}),
                (
                    ("--all",),
                    {"action": "store_true", "help": "download all models"},
                ),
            ],
        ),
        "chapters": (
            cmd_chapters,
            "list chapters in an epub file",
            [("runtime",)],
            [(("epub",), {"help": "path to epub file"})],
        ),
        "extract": (
            cmd_extract,
            "extract chapter text from epub",
            [("runtime",)],
            [
                (("epub",), {"help": "path to epub file"}),
                (("-o", "--output"), {"required": True, "help": "output workdir"}),
            ],
        ),
        "dramatize": (
            cmd_dramatize,
            "run full dramatization pipeline",
            [
                ("paths",),
                ("scripting",),
                ("chapter_selection",),
                ("tts_engine",),
                ("cast",),
                ("export",),
                ("runtime",),
                ("pipeline",),
            ],
            [
                (
                    ("--revise",),
                    {
                        "action": "store_true",
                        "help": "review chunks during generation and retry on failure",
                    },
                ),
                (
                    ("--retake",),
                    {
                        "action": "store_true",
                        "help": "validate segment wavs inline; re-synthesize bad takes",
                    },
                ),
                (
                    ("--callback",),
                    {
                        "action": "store_true",
                        "help": "validate audition/emote wavs inline; re-emote bad takes",
                    },
                ),
                (
                    ("--strict",),
                    {
                        "action": "store_true",
                        "help": "enable all validation checks (--revise --retake --callback)",
                    },
                ),
                (
                    ("--emotions",),
                    {
                        "action": "store_true",
                        "help": "also run the emote phase (per-emotion voice variants); "
                        "disabled by default",
                    },
                ),
                (
                    ("--preset-voices",),
                    {
                        "action": "store_true",
                        "help": "use preset backend voices for audition; perform then "
                        "uses voice ids + emotion instructions (no cloning)",
                    },
                ),
                (
                    ("--directed",),
                    {
                        "action": "store_true",
                        "help": "interactive casting loop during audition "
                        "(combine with --preset-voices)",
                    },
                ),
            ],
        ),
        "cast": (
            _cmd_cast,
            "generate cast list from book text",
            [("scripting",), ("chapter_selection",), ("cast",), ("runtime",)],
            [(("workdir",), {"help": "path to workdir"})],
        ),
        "design": (
            _cmd_design,
            "add or update a character in the cast",
            [("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (("--name",), {"required": True, "help": "character name"}),
                (("--text",), {"help": "audition line (text to speak)"}),
                (("--description",), {"help": "voice description (prompt)"}),
            ],
        ),
        "audition": (
            _cmd_audition,
            "generate per-character base voice (description only)",
            [("tts_engine",), ("cast",), ("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (
                    ("--audition-line",),
                    {"help": "override audition line for all characters"},
                ),
                (
                    ("--callback",),
                    {
                        "action": "store_true",
                        "help": "validate new voice wavs inline and re-emote bad takes",
                    },
                ),
                (
                    ("--preset-voices",),
                    {
                        "action": "store_true",
                        "help": "use preset voices from the http backend (requires --api-base)",
                    },
                ),
                (
                    ("--directed",),
                    {
                        "action": "store_true",
                        "help": "interactive casting loop: audition voices and pick one "
                        "per character (combine with --preset-voices)",
                    },
                ),
            ],
        ),
        "emote": (
            _cmd_emote,
            "generate per-emotion voice variants for each character",
            [("tts_engine",), ("cast",), ("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (
                    ("--audition-line",),
                    {"help": "override audition line for all characters"},
                ),
                (
                    ("--callback",),
                    {
                        "action": "store_true",
                        "help": "validate new voice wavs inline and re-emote bad takes",
                    },
                ),
            ],
        ),
        "callback": (
            _cmd_callback,
            "review and re-emote corrupted audition/emote wavs",
            [("tts_engine",), ("cast",), ("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (
                    ("-n", "--dry-run"),
                    {
                        "action": "store_true",
                        "help": "report only; don't delete or regenerate",
                    },
                ),
                (
                    ("--prune",),
                    {
                        "action": "store_true",
                        "help": "delete offenders but skip regeneration",
                    },
                ),
            ],
        ),
        "script": (
            _cmd_script,
            "dramatize chapters into scripts",
            [("scripting",), ("chapter_selection",), ("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (
                    ("--revise",),
                    {
                        "action": "store_true",
                        "help": "review chunks during generation and retry on failure",
                    },
                ),
            ],
        ),
        "perform": (
            _cmd_perform,
            "synthesize audio from dramatized scripts",
            [("chapter_selection",), ("tts_engine",), ("cast",), ("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (
                    ("--retake",),
                    {
                        "action": "store_true",
                        "help": "validate segment wavs inline; re-synthesize bad takes",
                    },
                ),
            ],
        ),
        "revise": (
            _cmd_revise,
            "review and repair scripts (missing / hallucinated segments)",
            [("scripting",), ("chapter_selection",), ("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (
                    ("-n", "--dry-run"),
                    {
                        "action": "store_true",
                        "help": "report only; don't modify scripts",
                    },
                ),
                (
                    ("--prune",),
                    {
                        "action": "store_true",
                        "help": "strip hallucinated segments but skip LLM fix-missing",
                    },
                ),
            ],
        ),
        "synthesize": (
            cmd_synthesize,
            "convert text files to wav audio",
            [("chapter_selection",), ("delivery",), ("tts_engine",), ("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (
                    ("--retake",),
                    {
                        "action": "store_true",
                        "help": "validate segment wavs inline; re-synthesize bad takes",
                    },
                ),
            ],
        ),
        "export": (
            cmd_export,
            "convert wav files to mp3",
            [("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (("-o", "--output"), {"help": "output directory"}),
                (
                    ("-b", "--bitrate"),
                    {"default": DEFAULT_BITRATE, "help": "mp3 bitrate"},
                ),
                (
                    ("--m4b",),
                    {"action": "store_true", "help": "export as m4b audiobook"},
                ),
            ],
        ),
        "locate": (
            cmd_locate,
            "look up the segment wav at an audio time position",
            [("runtime",)],
            [
                (("wav",), {"help": "path to a chapter wav"}),
                (("time",), {"help": "time position (seconds or m:ss.sss)"}),
            ],
        ),
        "retake": (
            _cmd_retake,
            "review and regenerate corrupted segment wavs",
            [
                ("chapter_selection",),
                ("tts_engine",),
                ("cast",),
                ("runtime",),
            ],
            [
                (("workdir",), {"help": "path to workdir"}),
                (
                    ("--command",),
                    {
                        "default": "perform",
                        "choices": ["perform", "synthesize"],
                        "help": "which pipeline's segments to scan (default: perform)",
                    },
                ),
                (
                    ("-n", "--dry-run"),
                    {
                        "action": "store_true",
                        "help": "report only; don't delete or regenerate",
                    },
                ),
                (
                    ("--prune",),
                    {
                        "action": "store_true",
                        "help": "delete offenders but skip regeneration",
                    },
                ),
            ],
        ),
        "clean": (
            cmd_clean,
            "remove intermediate chunk files",
            [("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (("-n", "--dry-run"), {"action": "store_true", "help": "dry run"}),
            ],
        ),
        "convert": (
            cmd_convert,
            "run full conversion pipeline",
            [
                ("paths",),
                ("export",),
                ("chapter_selection",),
                ("delivery",),
                ("runtime",),
                ("tts_engine",),
                ("pipeline",),
            ],
            [
                (
                    ("--retake",),
                    {
                        "action": "store_true",
                        "help": "validate segment wavs inline; re-synthesize bad takes",
                    },
                ),
                (
                    ("--strict",),
                    {
                        "action": "store_true",
                        "help": "enable all validation checks (--retake)",
                    },
                ),
            ],
        ),
    }

    for name, (func, help_text, groups, args_list) in commands.items():
        p = subparsers.add_parser(name, help=help_text)
        for arg_args, arg_kwargs in args_list:
            p.add_argument(*arg_args, **arg_kwargs)
        for g in groups:
            add_common_args(p, group=g[0])
        p.set_defaults(func=func)

    args = parser.parse_args()
    from .config import DEFAULT_SEED

    origin = "env" if os.getenv("AUTIOBOOK_SEED") else "random"
    print(f"seed: {DEFAULT_SEED} ({origin}); set AUTIOBOOK_SEED to reproduce")
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted!")
        sys.exit(1)
    except (RuntimeError, FileNotFoundError, OSError) as e:
        # ValidationError (subclass of RuntimeError) prints its own summary
        try:
            from .dramatize import ValidationError

            if isinstance(e, ValidationError):
                sys.exit(1)
        except ImportError:
            pass
        print(f"error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
