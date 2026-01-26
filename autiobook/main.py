"""cli entry point for autiobook."""

import argparse
import sys
from pathlib import Path

from .config import (
    BASE_MODEL,
    DEFAULT_BITRATE,
    DEFAULT_MODEL,
    VOICE_DESIGN_MODEL,
)
from .design import cmd_design
from .dramatize import (
    cmd_audition,
    cmd_cast,
    cmd_fix,
    cmd_perform,
    cmd_script,
    cmd_validate,
    dramatize_book,
)
from .epub import ensure_extracted, parse_epub
from .export import export_audiobook
from .showcase import cmd_showcase
from .tts import synthesize_chapters
from .utils import add_common_args, get_chapters, get_pipeline_paths, get_tts_config


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


def _run_pipeline(args, process_fn, name):
    """common helper for full pipelines."""
    epub_path, workdir = get_pipeline_paths(args)
    audiobook_dir = workdir / "audiobook"
    chapters = get_chapters(args)

    ensure_extracted(epub_path, workdir, force=args.force)

    config = get_tts_config(args)
    process_fn(workdir, config, chapters)

    print(f"export: exporting chapters to {audiobook_dir}/...")
    new, skipped = export_audiobook(
        workdir, audiobook_dir, args.bitrate, force=args.force
    )

    msg = f"{name}: done - {new} chapter(s) exported"
    if skipped > 0:
        msg += f" ({skipped} skipped)"
    print(msg)


def cmd_dramatize(args):
    """generate script and cast using LLM."""

    def process_fn(workdir, config, chapters):
        print(f"dramatize: dramatizing chapters in {workdir}...")
        dramatize_book(
            workdir,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            chapters=chapters,
            tts_config=config,
            pooled=args.pooled,
            verbose=args.verbose,
            force=args.force,
        )

    _run_pipeline(args, process_fn, "dramatize")


def cmd_convert(args):
    """run full conversion pipeline."""

    def process_fn(workdir, config, chapters):
        print(f"synthesize: synthesizing chapters in {workdir}/synthesize/...")
        synthesize_chapters(
            workdir, config, chapters, args.instruct, args.pooled, force=args.force
        )

    _run_pipeline(args, process_fn, "convert")


def cmd_synthesize(args):
    """convert text files to wav audio."""
    workdir = Path(args.workdir)
    chapters = get_chapters(args)
    config = get_tts_config(args)

    print(f"synthesize: synthesizing chapters in {workdir}/synthesize/...")
    synthesize_chapters(
        workdir, config, chapters, args.instruct, args.pooled, force=args.force
    )

    print("synthesize: done")


def cmd_export(args):
    """convert wav files to mp3 with metadata."""
    workdir = Path(args.workdir)
    output_dir = Path(args.output) if args.output else workdir / "audiobook"

    print(f"export: exporting chapters to {output_dir}/...")
    new, skipped = export_audiobook(workdir, output_dir, args.bitrate, force=args.force)

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
            ],
            [],
        ),
        "cast": (
            cmd_cast,
            "generate cast list from book text",
            [("scripting",), ("chapter_selection",), ("cast",), ("runtime",)],
            [(("workdir",), {"help": "path to workdir"})],
        ),
        "design": (
            cmd_design,
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
            cmd_audition,
            "generate character voice samples",
            [("cast",), ("runtime",)],
            [(("workdir",), {"help": "path to workdir"})],
        ),
        "showcase": (
            cmd_showcase,
            "generate emotion samples for character voices",
            [("cast",), ("runtime",), ("delivery",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (("--text",), {"help": "custom text to speak"}),
                (
                    ("--emotion",),
                    {"action": "append", "help": "filter emotions to showcase"},
                ),
            ],
        ),
        "script": (
            cmd_script,
            "dramatize chapters into scripts",
            [("scripting",), ("chapter_selection",), ("runtime",)],
            [(("workdir",), {"help": "path to workdir"})],
        ),
        "perform": (
            cmd_perform,
            "synthesize audio from dramatized scripts",
            [("chapter_selection",), ("tts_engine",), ("cast",), ("runtime",)],
            [(("workdir",), {"help": "path to workdir"})],
        ),
        "validate": (
            cmd_validate,
            "verify scripts match source text",
            [("chapter_selection",), ("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (
                    ("--missing",),
                    {"action": "store_true", "help": "check missing text"},
                ),
                (
                    ("--hallucinated",),
                    {"action": "store_true", "help": "check hallucinated segments"},
                ),
            ],
        ),
        "fix": (
            cmd_fix,
            "fix script issues",
            [("scripting",), ("chapter_selection",), ("runtime",)],
            [
                (("workdir",), {"help": "path to workdir"}),
                (("--missing",), {"action": "store_true", "help": "fill missing"}),
                (
                    ("--hallucinated",),
                    {"action": "store_true", "help": "remove hallucinated"},
                ),
                (("--context-chars",), {"type": int, "help": "chars of context"}),
                (
                    ("--context-paragraphs",),
                    {"type": int, "help": "paragraphs of context"},
                ),
            ],
        ),
        "synthesize": (
            cmd_synthesize,
            "convert text files to wav audio",
            [("chapter_selection",), ("delivery",), ("tts_engine",), ("runtime",)],
            [(("workdir",), {"help": "path to workdir"})],
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
            ],
            [],
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
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted!")
        sys.exit(1)


if __name__ == "__main__":
    main()
