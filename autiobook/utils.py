"""utility functions."""

import argparse
from pathlib import Path
from typing import Iterator

from .config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_SPEAKER,
    MAX_CHUNK_SIZE,
    TXT_EXT,
    WAV_EXT,
)


def parse_chapter_range(spec: str) -> list[int]:
    """parse chapter range spec like '1-5' or '1,3,5' into list of ints."""
    chapters = []
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
    if group in ["all", "chapters"]:
        parser.add_argument("-c", "--chapters", help="chapter range (e.g., 1-5, 1,3,5)")

    if group in ["all", "tts"]:
        parser.add_argument(
            "--batch-size", type=int, default=64, help="batch size for tts generation"
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=MAX_CHUNK_SIZE,
            help="max chars per chunk (smaller = faster)",
        )
        parser.add_argument(
            "--no-compile", action="store_true", help="disable torch.compile optimization"
        )
        parser.add_argument("--no-warmup", action="store_true", help="skip model warmup")
        parser.add_argument(
            "--pooled",
            action="store_true",
            help="pool chunks across chapters for better batch utilization",
        )
        parser.add_argument(
            "--greedy", action="store_true", help="use greedy decoding (faster, less varied)"
        )
        parser.add_argument(
            "--temperature", type=float, default=0.9, help="sampling temperature (lower = faster)"
        )

    if group in ["all", "speaker"]:
        parser.add_argument("-s", "--speaker", default=DEFAULT_SPEAKER, help="tts voice")

    if group in ["all", "instruct"]:
        parser.add_argument("-i", "--instruct", help="instruction for tts (string or file path)")

    if group in ["all", "llm"]:
        parser.add_argument("--api-base", help="openai api base url")
        parser.add_argument("--api-key", help="openai api key")
        parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="llm model name")

    if group in ["all", "cast"]:
        parser.add_argument(
            "--min-appearances",
            type=int,
            default=0,
            help="minimum appearances for dedicated voice (others use generic Extra voices)",
        )

    if group in ["all", "logging"]:
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="enable verbose logging"
        )


def iter_pending_chapters(
    workdir: Path,
    chapters: list[int] | None = None,
    source_ext: str = TXT_EXT,
    target_ext: str = WAV_EXT,
    skip_message: str = "already exists",
) -> Iterator[tuple[Path, Path]]:
    """yield (source_path, target_path) for chapters that need processing."""
    source_files = sorted(workdir.glob(f"*{source_ext}"))

    for source_path in source_files:
        target_path = source_path.with_suffix(target_ext)

        try:
            chapter_num = int(source_path.stem.split("_")[0])
        except ValueError:
            continue

        if chapters and chapter_num not in chapters:
            continue

        if target_path.exists():
            print(f"skipping {source_path.name} ({skip_message})")
            continue

        yield source_path, target_path
