"""utility functions."""

import argparse
from pathlib import Path

from .config import (
    BASE_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    MAX_CHUNK_SIZE,
)


def parse_chapter_range(spec: str) -> list[int]:
    """parse chapter range spec like '1-5' or '1,3,5' into list of ints."""
    chapters: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            chapters.extend(range(int(start), int(end) + 1))
        else:
            chapters.append(int(part))
    return chapters


def add_common_args(parser: argparse.ArgumentParser, group: str = "all"):
    """add common arguments to parser."""
    g = parser.add_argument_group(group) if group != "all" else parser

    if group in ["all", "chapter_selection"]:
        g.add_argument("-c", "--chapters", help="chapter range (e.g., 1-5, 1,3,5)")

    if group in ["all", "tts_engine"]:
        g.add_argument(
            "--batch-size", type=int, default=64, help="batch size for tts generation"
        )
        g.add_argument(
            "--chunk-size",
            type=int,
            default=MAX_CHUNK_SIZE,
            help="max chars per chunk (smaller = faster)",
        )
        g.add_argument(
            "--no-compile",
            action="store_true",
            help="disable torch.compile optimization",
        )
        g.add_argument("--no-warmup", action="store_true", help="skip model warmup")
        g.add_argument(
            "--pooled",
            action="store_true",
            help="pool chunks across chapters for better batch utilization",
        )
        g.add_argument(
            "--greedy",
            action="store_true",
            help="use greedy decoding (faster, less varied)",
        )
        g.add_argument(
            "--temperature",
            type=float,
            default=0.9,
            help="sampling temperature (lower = faster)",
        )

    if group in ["all", "delivery"]:
        g.add_argument(
            "-s",
            "--speaker",
            default=DEFAULT_SPEAKER,
            help="base tts speaker",
        )
        g.add_argument(
            "--voice",
            help="cloned voice (name from audition/)",
        )
        g.add_argument(
            "-i", "--instruct", help="instruction for tts (string or file path)"
        )

    if group in ["all", "scripting"]:
        g.add_argument("--api-base", help="openai api base url")
        g.add_argument("--api-key", help="openai api key")
        g.add_argument("--model", default=DEFAULT_LLM_MODEL, help="llm model name")

    if group in ["all", "runtime"]:
        g.add_argument(
            "-v", "--verbose", action="store_true", help="enable verbose logging"
        )
        g.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="ignore resume state and force processing",
        )

    if group in ["all", "paths"]:
        g.add_argument("epub", help="path to epub file")
        g.add_argument(
            "-o",
            "--output",
            help="workdir for intermediate files (default: <epub>_output/)",
        )

    if group in ["all", "export"]:
        from .config import DEFAULT_BITRATE

        g.add_argument("-b", "--bitrate", default=DEFAULT_BITRATE, help="mp3 bitrate")


def get_pipeline_paths(args) -> tuple[Path, Path]:
    """get epub_path and workdir from args, inferring if needed."""
    epub_path = Path(args.epub)
    if args.output:
        workdir = Path(args.output)
    else:
        # infer workdir: /path/to/book.epub -> /path/to/book_output/
        workdir = epub_path.parent / (epub_path.stem + "_output")

    return epub_path, workdir


def get_chapters(args) -> list[int] | None:
    """extract chapter list from args."""
    if args.chapters:
        return parse_chapter_range(args.chapters)
    return None


def get_tts_config(args):
    """extract tts config from args."""
    from .tts import TTSConfig

    model_name = DEFAULT_MODEL
    # use base model for voice cloning if voice is specified
    if hasattr(args, "voice") and args.voice:
        model_name = BASE_MODEL

    config = TTSConfig(
        model_name=model_name,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        compile_model=not args.no_compile,
        warmup=not args.no_warmup,
        do_sample=not args.greedy,
        temperature=args.temperature,
    )

    if hasattr(args, "speaker"):
        config.speaker = args.speaker

    if hasattr(args, "voice") and args.voice:
        setattr(config, "voice", args.voice)

    return config
