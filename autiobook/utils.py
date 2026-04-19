"""utility functions."""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List

from .config import (
    BASE_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    DEFAULT_THINKING_BUDGET,
    LOG_FILE,
    MAX_CHUNK_SIZE,
    VOICE_DESIGN_MODEL,
)


class Logger:
    """simple file logger for debugging."""

    _instance: "Logger | None" = None
    _workdir: Path | None = None

    @classmethod
    def init(cls, workdir: Path) -> "Logger":
        """initialize logger with workdir."""
        cls._workdir = workdir
        cls._instance = cls()
        return cls._instance

    @classmethod
    def get(cls) -> "Logger | None":
        """get current logger instance."""
        return cls._instance

    def _log_path(self) -> Path | None:
        if self._workdir is None:
            return None
        return self._workdir / LOG_FILE

    def log(self, category: str, message: str, data: dict[str, Any] | None = None):
        """log a message with optional structured data."""
        path = self._log_path()
        if path is None:
            return

        timestamp = datetime.now().isoformat(timespec="seconds")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"[{timestamp}] {category}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"{message}\n")
            if data:
                for key, value in data.items():
                    f.write(f"\n--- {key} ---\n")
                    if isinstance(value, str):
                        f.write(f"{value}\n")
                    else:
                        f.write(f"{value!r}\n")


def log(category: str, message: str, data: dict[str, Any] | None = None):
    """convenience function to log if logger is initialized."""
    logger = Logger.get()
    if logger:
        logger.log(category, message, data)


def dir_mtime(path: Path) -> float:
    """get latest mtime of any file in directory, or 0 if empty/missing."""
    if not path.exists():
        return 0
    latest = 0.0
    for f in path.rglob("*"):
        if f.is_file():
            latest = max(latest, f.stat().st_mtime)
    return latest


SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """split text into chunks at sentence boundaries, force-splitting if needed."""
    sentences = SENTENCE_ENDINGS.split(text)

    chunks = []
    current_chunk: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_len = len(sentence)

        if current_length + sentence_len > max_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        if sentence_len > max_size:
            words = sentence.split(" ")
            current_word_chunk: List[str] = []
            current_word_len = 0

            for word in words:
                word_len = len(word)
                if current_word_len + word_len + 1 > max_size:
                    chunks.append(" ".join(current_word_chunk))
                    current_word_chunk = []
                    current_word_len = 0
                current_word_chunk.append(word)
                current_word_len += word_len + 1

            if current_word_chunk:
                remainder = " ".join(current_word_chunk)
                current_chunk.append(remainder)
                current_length += len(remainder) + 1
            continue

        current_chunk.append(sentence)
        current_length += sentence_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


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
        from .tts import _HAS_LOCAL

        g.add_argument(
            "--tts-model",
            default="",
            help="default tts model (used unless overridden by --tts-*-model)",
        )
        g.add_argument(
            "--tts-design-model",
            default="",
            help="tts model for voice design (emote)",
        )
        g.add_argument(
            "--tts-clone-model",
            default="",
            help="tts model for voice cloning (perform)",
        )
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
            "--pooled",
            action="store_true",
            help="pool chunks across chapters for better batch utilization",
        )
        g.add_argument(
            "--temperature",
            type=float,
            default=None,
            help="sampling temperature (lower = faster; backend default if unset)",
        )
        if _HAS_LOCAL:
            g.add_argument(
                "--no-compile",
                action="store_true",
                help="disable torch.compile optimization",
            )
            g.add_argument("--no-warmup", action="store_true", help="skip model warmup")
            g.add_argument(
                "--greedy",
                action="store_true",
                help="use greedy decoding (faster, less varied)",
            )

    if group in ["all", "delivery"]:
        g.add_argument(
            "-s",
            "--voice",
            default=DEFAULT_SPEAKER,
            help="base tts voice",
        )
        g.add_argument(
            "--speaker",
            help="cloned speaker (name from audition/)",
        )
        g.add_argument(
            "-i", "--instruct", help="instruction for tts (string or file path)"
        )

    if group in ["all", "tts_engine", "scripting"]:
        # avoid duplicate registration when both groups are present
        if not any(
            a.option_strings and "--api-base" in a.option_strings
            for a in parser._actions
        ):
            g.add_argument(
                "--api-base",
                default=os.getenv("OPENAI_BASE_URL"),
                help="api base url (used for both llm and tts; "
                "defaults to $OPENAI_BASE_URL)",
            )
            g.add_argument(
                "--api-key",
                default=os.getenv("OPENAI_API_KEY"),
                help="api key (defaults to $OPENAI_API_KEY)",
            )

    if group in ["all", "scripting"]:
        g.add_argument(
            "--llm-model",
            "--model",
            default=DEFAULT_LLM_MODEL,
            dest="model",
            help="llm model name",
        )
        g.add_argument(
            "--thinking-budget",
            type=int,
            default=DEFAULT_THINKING_BUDGET,
            help="tokens for extended thinking (0 = unlimited)",
        )

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

    if group in ["all", "pipeline"]:
        g.add_argument(
            "--step",
            action="store_true",
            help="exit after each pipeline phase (re-run to continue)",
        )
        g.add_argument(
            "--redo",
            action="store_true",
            help="re-run the last completed pipeline phase",
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
        g.add_argument(
            "--m4b", action="store_true", help="export as m4b audiobook with metadata"
        )


def find_redo_phase(workdir: Path, phases: list[str]) -> str | None:
    """find the last completed phase to redo.

    scans phases in order, returns the last one with output files.
    returns None if no phases have completed.
    """
    last_complete = None
    for phase in phases:
        phase_dir = workdir / phase
        if phase_dir.exists() and any(phase_dir.iterdir()):
            last_complete = phase
    return last_complete


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
    spec = getattr(args, "chapters", None)
    if spec:
        return parse_chapter_range(spec)
    return None


def create_tts_engine(config):
    """create the appropriate tts engine based on config type."""
    from .tts_http import HTTPTTSConfig

    if isinstance(config, HTTPTTSConfig):
        from .tts_http import HTTPTTSEngine

        return HTTPTTSEngine(config)

    from .tts import TTSEngine

    return TTSEngine(config)


def _resolve_tts_model(args, default: str, override_flag: str = "") -> str:
    """resolve tts model name from args with fallback chain.

    priority: --tts-{override}-model > --tts-model > default
    """
    if override_flag:
        override = getattr(args, f"tts_{override_flag}_model", "")
        if override:
            return override
    base = getattr(args, "tts_model", "")
    return base or default


def _build_http_config(args, model: str):
    """build http tts config from args."""
    from .tts_http import HTTPTTSConfig

    config = HTTPTTSConfig(
        api_base=args.api_base,
        model=model,
        chunk_size=getattr(args, "chunk_size", MAX_CHUNK_SIZE),
        temperature=getattr(args, "temperature", None),
    )
    if hasattr(args, "voice"):
        config.speaker = args.voice
    if hasattr(args, "speaker") and args.speaker:
        setattr(config, "voice", args.speaker)
    return config


def _build_local_config(args, model: str):
    """build local tts config from args."""
    from .tts import TTSConfig

    config = TTSConfig(
        model_name=model,
        batch_size=getattr(args, "batch_size", 64),
        chunk_size=getattr(args, "chunk_size", MAX_CHUNK_SIZE),
        compile_model=not getattr(args, "no_compile", False),
        warmup=not getattr(args, "no_warmup", False),
        do_sample=not getattr(args, "greedy", False),
        temperature=getattr(args, "temperature", None),
    )
    if hasattr(args, "voice"):
        config.speaker = args.voice
    if hasattr(args, "speaker") and args.speaker:
        setattr(config, "voice", args.speaker)
    return config


def get_tts_config(args):
    """extract tts config for synthesis/general use."""
    default = (
        BASE_MODEL if (hasattr(args, "speaker") and args.speaker) else DEFAULT_MODEL
    )
    model = _resolve_tts_model(args, default)

    if getattr(args, "api_base", None):
        return _build_http_config(args, model)
    return _build_local_config(args, model)


def get_design_config(args):
    """extract tts config for voice design (audition)."""
    model = getattr(args, "tts_design_model", "") or VOICE_DESIGN_MODEL

    if getattr(args, "api_base", None):
        return _build_http_config(args, model)
    return _build_local_config(args, model)


def get_clone_config(args):
    """extract tts config for voice cloning (perform)."""
    model = getattr(args, "tts_clone_model", "") or BASE_MODEL

    if getattr(args, "api_base", None):
        return _build_http_config(args, model)
    return _build_local_config(args, model)
