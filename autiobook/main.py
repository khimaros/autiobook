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
from .dramatize import (
    cmd_audition,
    cmd_cast,
    cmd_fix,
    cmd_perform,
    cmd_script,
    cmd_validate,
    dramatize_book,
)
from .epub import parse_epub, save_extracted
from .export import export_audiobook
from .tts import synthesize_chapters
from .utils import add_common_args, get_chapters, get_tts_config


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

    print(f"extract: parsing {epub_path.name}...")
    book, cover_data = parse_epub(epub_path)

    print(f"extract: extracting {len(book.chapters)} chapters to {workdir}/...")
    save_extracted(book, workdir, cover_data)

    print("extract: done")


def cmd_dramatize(args):
    """generate script and cast using LLM."""
    workdir = Path(args.workdir)
    chapters = get_chapters(args)
    tts_config = get_tts_config(args)

    print(f"dramatize: dramatizing chapters in {workdir}/...")
    dramatize_book(
        workdir,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
        tts_config=tts_config,
        pooled=args.pooled,
        verbose=args.verbose,
        force=args.force,
    )

    print("dramatize: done")


def cmd_synthesize(args):
    """convert text files to wav audio."""
    workdir = Path(args.workdir)
    chapters = get_chapters(args)
    config = get_tts_config(args)

    print(f"synthesize: synthesizing chapters in {workdir}/...")
    synthesize_chapters(
        workdir, config, chapters, args.instruct, args.pooled, force=args.force
    )

    print("synthesize: done")


def cmd_export(args):
    """convert wav files to mp3 with metadata."""
    workdir = Path(args.workdir)
    output_dir = Path(args.output)

    print(f"export: exporting chapters to {output_dir}/...")
    count = export_audiobook(workdir, output_dir, args.bitrate, force=args.force)

    print(f"export: {count} chapter(s) exported")


def cmd_clean(args):
    """remove intermediate files (chunks directories)."""
    import shutil

    from .config import CHUNKS_DIR

    workdir = Path(args.workdir)
    chunks_dir = workdir / CHUNKS_DIR

    if not chunks_dir.exists():
        print("clean: no chunks directory found")
        return

    # count subdirectories
    chunk_dirs = list(chunks_dir.iterdir()) if chunks_dir.is_dir() else []
    count = len([d for d in chunk_dirs if d.is_dir()])

    if args.dry_run:
        print(f"clean: would remove {count} chunk directories from {chunks_dir}")
        return

    shutil.rmtree(chunks_dir)
    print(f"clean: removed {count} chunk directories")


def cmd_convert(args):
    """run full conversion pipeline."""
    epub_path = Path(args.epub)
    workdir = Path(args.output)
    audiobook_dir = Path(args.audiobook) if args.audiobook else workdir / "audiobook"
    chapters = get_chapters(args)

    # extract
    print(f"extract: parsing {epub_path.name}...")
    book, cover_data = parse_epub(epub_path)
    print(f"extract: extracting {len(book.chapters)} chapters to {workdir}/...")
    save_extracted(book, workdir, cover_data)

    # synthesize
    config = get_tts_config(args)
    print("synthesize: synthesizing chapters...")
    synthesize_chapters(
        workdir, config, chapters, args.instruct, args.pooled, force=args.force
    )

    # export
    print(f"export: exporting chapters to {audiobook_dir}/...")
    count = export_audiobook(workdir, audiobook_dir, args.bitrate, force=args.force)

    print(f"convert: done - {count} chapter(s) exported")


def main():
    parser = argparse.ArgumentParser(
        prog="autiobook",
        description="convert epub files to audiobooks using qwen3-tts",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # download command
    p_download = subparsers.add_parser("download", help="download tts model weights")
    p_download.add_argument("-m", "--model", default=DEFAULT_MODEL, help="model name")
    p_download.add_argument(
        "--all", action="store_true", help="download all models (custom, design, base)"
    )
    add_common_args(p_download, group="logging")
    p_download.set_defaults(func=cmd_download)

    # chapters command
    p_chapters = subparsers.add_parser("chapters", help="list chapters in an epub file")
    p_chapters.add_argument("epub", help="path to epub file")
    add_common_args(p_chapters, group="logging")
    p_chapters.set_defaults(func=cmd_chapters)

    # extract command
    p_extract = subparsers.add_parser("extract", help="extract chapter text from epub")
    p_extract.add_argument("epub", help="path to epub file")
    p_extract.add_argument("-o", "--output", required=True, help="output workdir")
    add_common_args(p_extract, group="logging")
    p_extract.set_defaults(func=cmd_extract)

    # dramatize command
    p_dramatize = subparsers.add_parser(
        "dramatize", help="run full dramatization pipeline"
    )
    p_dramatize.add_argument("workdir", help="path to workdir")
    add_common_args(p_dramatize, group="llm")
    add_common_args(p_dramatize, group="chapters")
    add_common_args(p_dramatize, group="tts")
    add_common_args(p_dramatize, group="cast")
    add_common_args(p_dramatize, group="logging")
    p_dramatize.set_defaults(func=cmd_dramatize)

    # cast command
    p_cast = subparsers.add_parser("cast", help="generate cast list from book text")
    p_cast.add_argument("workdir", help="path to workdir")
    add_common_args(p_cast, group="llm")
    add_common_args(p_cast, group="chapters")
    add_common_args(p_cast, group="cast")
    add_common_args(p_cast, group="logging")
    p_cast.set_defaults(func=cmd_cast)

    # audition command
    p_audition = subparsers.add_parser(
        "audition", help="generate character voice samples"
    )
    p_audition.add_argument("workdir", help="path to workdir")
    add_common_args(p_audition, group="cast")
    add_common_args(p_audition, group="logging")
    p_audition.set_defaults(func=cmd_audition)

    # script command
    p_script = subparsers.add_parser("script", help="dramatize chapters into scripts")
    p_script.add_argument("workdir", help="path to workdir")
    add_common_args(p_script, group="llm")
    add_common_args(p_script, group="chapters")
    add_common_args(p_script, group="logging")
    p_script.set_defaults(func=cmd_script)

    # perform command
    p_perform = subparsers.add_parser(
        "perform", help="synthesize audio from dramatized scripts"
    )
    p_perform.add_argument("workdir", help="path to workdir")
    add_common_args(p_perform, group="chapters")
    add_common_args(p_perform, group="tts")
    add_common_args(p_perform, group="cast")
    add_common_args(p_perform, group="logging")
    p_perform.set_defaults(func=cmd_perform)

    # validate command
    p_validate = subparsers.add_parser(
        "validate", help="verify scripts match source text"
    )
    p_validate.add_argument("workdir", help="path to workdir")
    p_validate.add_argument(
        "--missing", action="store_true", help="check for missing text"
    )
    p_validate.add_argument(
        "--hallucinated", action="store_true", help="check for hallucinated segments"
    )
    add_common_args(p_validate, group="chapters")
    add_common_args(p_validate, group="logging")
    p_validate.set_defaults(func=cmd_validate)

    # fix command
    p_fix = subparsers.add_parser(
        "fix", help="fix script issues (fill missing, remove hallucinated)"
    )
    p_fix.add_argument("workdir", help="path to workdir")
    p_fix.add_argument(
        "--missing", action="store_true", help="fill missing text segments"
    )
    p_fix.add_argument(
        "--hallucinated", action="store_true", help="remove hallucinated segments"
    )
    p_fix.add_argument(
        "--context-chars",
        type=int,
        metavar="N",
        help="characters of context before/after missing text (default: 500)",
    )
    p_fix.add_argument(
        "--context-paragraphs",
        type=int,
        metavar="N",
        help="paragraphs of context before/after missing text (alternative to --context-chars)",
    )
    add_common_args(p_fix, group="llm")
    add_common_args(p_fix, group="chapters")
    add_common_args(p_fix, group="logging")
    p_fix.set_defaults(func=cmd_fix)

    # synthesize command
    p_synth = subparsers.add_parser(
        "synthesize", help="convert text files to wav audio"
    )
    p_synth.add_argument("workdir", help="path to workdir with txt files")
    add_common_args(p_synth, group="chapters")
    add_common_args(p_synth, group="speaker")
    add_common_args(p_synth, group="instruct")
    add_common_args(p_synth, group="tts")
    add_common_args(p_synth, group="logging")
    p_synth.set_defaults(func=cmd_synthesize)

    # export command
    p_export = subparsers.add_parser("export", help="convert wav files to mp3")
    p_export.add_argument("workdir", help="path to workdir with wav files")
    p_export.add_argument(
        "-o", "--output", required=True, help="output directory for mp3 files"
    )
    p_export.add_argument(
        "-b", "--bitrate", default=DEFAULT_BITRATE, help="mp3 bitrate"
    )
    add_common_args(p_export, group="logging")
    p_export.set_defaults(func=cmd_export)

    # clean command
    p_clean = subparsers.add_parser("clean", help="remove intermediate chunk files")
    p_clean.add_argument("workdir", help="path to workdir")
    p_clean.add_argument(
        "-n", "--dry-run", action="store_true", help="show what would be removed"
    )
    add_common_args(p_clean, group="logging")
    p_clean.set_defaults(func=cmd_clean)

    # convert command (full pipeline)
    p_convert = subparsers.add_parser("convert", help="run full conversion pipeline")
    p_convert.add_argument("epub", help="path to epub file")
    p_convert.add_argument(
        "-o", "--output", required=True, help="workdir for intermediate files"
    )
    p_convert.add_argument("--audiobook", help="output directory for mp3 files")
    p_convert.add_argument(
        "-b", "--bitrate", default=DEFAULT_BITRATE, help="mp3 bitrate"
    )
    add_common_args(p_convert, group="chapters")
    add_common_args(p_convert, group="speaker")
    add_common_args(p_convert, group="instruct")
    add_common_args(p_convert, group="logging")
    add_common_args(p_convert, group="tts")
    p_convert.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nAborted!")
        sys.exit(1)


if __name__ == "__main__":
    main()
