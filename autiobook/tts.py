"""tts engine wrapper for qwen3-tts with rocm optimizations."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import soundfile as sf  # type: ignore
from tqdm import tqdm  # type: ignore

from .audio import concatenate_audio, get_segments_dir
from .config import (
    DEFAULT_MODEL,
    DEFAULT_SEED,
    DEFAULT_SPEAKER,
    MAX_CHUNK_SIZE,
    PARAGRAPH_PAUSE_MS,
    SAMPLE_RATE,
    TXT_EXT,
    WAV_EXT,
)
from .epub import load_metadata
from .pooling import AudioTask, process_audio_pipeline
from .resume import ResumeManager, compute_hash, get_command_dir, list_chapters
from .utils import chunk_text  # re-exported for backwards compat

try:
    import torch
    import transformers  # type: ignore

    transformers.logging.set_verbosity_error()
    _HAS_LOCAL = True
except ModuleNotFoundError:
    _HAS_LOCAL = False

# warmup text for model compilation
WARMUP_TEXT = "Hello, this is a warmup."


def _require_local():
    """raise a clear error if local tts dependencies are missing."""
    if not _HAS_LOCAL:
        raise RuntimeError(
            "local tts dependencies not installed. install with: pip install autiobook[local]"
        )


def get_default_device() -> str:
    """get default device (cuda/rocm if available, else cpu)."""
    if not _HAS_LOCAL:
        return "cpu"
    has_cuda = torch.cuda.is_available()
    has_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None

    if has_cuda or has_rocm:
        device_count = torch.cuda.device_count()
        if device_count > 0:
            device_type = "cuda"
        else:
            print(
                "WARNING: CUDA/ROCm libraries detected but no devices found. Fallback to CPU."
            )
            device_type = "cpu"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    print(f"autodetected device type: {device_type}")
    return device_type


def is_rocm() -> bool:
    """check if running on rocm."""
    if not _HAS_LOCAL:
        return False
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def setup_rocm_env():
    """set environment variables for optimal rocm performance."""
    if not is_rocm():
        return
    env = {
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
        "MIOPEN_WORKSPACE_MAX": "256000000",
        "MIOPEN_FIND_MODE": "FAST",
        "MIOPEN_USER_DB_PATH": os.path.expanduser("~/.cache/miopen"),
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)


@dataclass
class TTSConfig:
    """configuration for tts generation."""

    model_name: str = DEFAULT_MODEL
    speaker: str = DEFAULT_SPEAKER
    language: str = "English"
    device: str = field(default_factory=get_default_device)
    batch_size: int = 64
    chunk_size: int = MAX_CHUNK_SIZE
    compile_model: bool = False
    warmup: bool = True
    do_sample: bool = True
    temperature: float | None = None
    max_new_tokens: int = 2048
    seed: int = DEFAULT_SEED


class TTSEngine:
    """wrapper for qwen3-tts model with rocm optimizations."""

    def __init__(self, config: TTSConfig | None = None):
        _require_local()
        setup_rocm_env()
        self.config = config or TTSConfig()
        self._model = None
        self._loaded_model_name = None
        self._compiled = False

    def _load_model(self):
        """lazy load and optimize the tts model."""
        if self._model and self._loaded_model_name == self.config.model_name:
            return

        if self._model:
            print(f"unloading model {self._loaded_model_name}...")
            del self._model
            torch.cuda.empty_cache()

        from qwen_tts import Qwen3TTSModel  # type: ignore

        # determine dtype and attention implementation
        is_cuda = "cuda" in self.config.device
        dtype = torch.bfloat16 if is_cuda else torch.float32
        attn_impl = "sdpa" if is_rocm() or not is_cuda else "flash_attention_2"

        print(
            f"loading {self.config.model_name} on {self.config.device} ({dtype}, {attn_impl})..."
        )
        self._model = Qwen3TTSModel.from_pretrained(
            self.config.model_name,
            device_map=self.config.device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        self._loaded_model_name = self.config.model_name

        if self.config.compile_model and is_cuda:
            self._compile_model()
        if self.config.warmup:
            self._warmup()

    def _compile_model(self):
        """apply torch.compile to model components."""
        if self._compiled or not self._model:
            return
        print("compiling model...")
        try:
            self._model.model.talker = torch.compile(  # type: ignore
                self._model.model.talker, mode="reduce-overhead", fullgraph=False
            )
            self._compiled = True
        except Exception as e:
            print(f"warning: torch.compile failed ({e})")

    def _warmup(self):
        """warmup model with a short synthesis to trigger compilation."""
        print("warming up model...")
        try:
            if "VoiceDesign" in self.config.model_name:
                self.design_voice(WARMUP_TEXT, "neutral voice")
            elif "Base" not in self.config.model_name:
                self.synthesize(WARMUP_TEXT)
        except Exception as e:
            print(f"warning: warmup failed ({e})")

    def _run_inference(self, func_name: str, **kwargs) -> tuple[Any, int]:
        """helper to run inference with correct context and params."""
        self._load_model()
        func = getattr(self._model, func_name)
        kwargs.update(
            {
                "language": self.config.language,
                "non_streaming_mode": True,
                "do_sample": self.config.do_sample,
            }
        )
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature

        ctx: Any = nullcontext()
        if "cuda" in self.config.device and is_rocm():
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                # include MATH backend as fallback for mismatched head counts
                ctx = sdpa_kernel(
                    [
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.MATH,
                    ]
                )
            except ImportError:
                pass

        if self.config.seed > 0:
            torch.manual_seed(self.config.seed)

        with ctx, torch.inference_mode():
            return cast(tuple[Any, int], func(**kwargs))

    def synthesize(
        self, text: str | list[str], instruct: str = "", speaker: str | None = None
    ) -> tuple[np.ndarray | list[np.ndarray], int]:
        """synthesize speech from text using current model."""
        wavs, sr = self._run_inference(
            "generate_custom_voice",
            text=text,
            speaker=speaker or self.config.speaker,
            instruct=instruct,
            max_new_tokens=self.config.max_new_tokens,
        )

        if isinstance(text, str):
            return wavs[0], sr
        return wavs, sr

    def design_voice(self, text: str, instruct: str) -> tuple[np.ndarray, int]:
        """generate a voice design sample."""
        wavs, sr = self._run_inference(
            "generate_voice_design",
            text=text,
            instruct=instruct,
        )
        return wavs[0], sr

    def clone_voice(
        self,
        text: str | list[str],
        ref_audio: np.ndarray | tuple | str,
        ref_text: str,
    ) -> tuple[np.ndarray | list[np.ndarray], int]:
        """clone voice from reference audio."""
        # ensure ref_audio is a tuple (audio, sr) as required by qwen_tts
        if isinstance(ref_audio, (str, Path)):
            audio_data, audio_sr = sf.read(str(ref_audio))
            ref_audio = (audio_data, audio_sr)
        elif isinstance(ref_audio, np.ndarray):
            # assume default sample rate if raw array passed
            ref_audio = (ref_audio, SAMPLE_RATE)

        wavs, sr = self._run_inference(
            "generate_voice_clone",
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

        if isinstance(text, str):
            return wavs[0], sr
        return wavs, sr

    def _generate_long(
        self,
        text: str,
        generation_fn: Any,
        desc: str = "generating chunks",
    ) -> tuple[np.ndarray, int]:
        """helper for generating long audio by chunking."""
        chunks = chunk_text(text, self.config.chunk_size)
        chunks = [c for c in chunks if c.strip()]

        if not chunks:
            return np.array([], dtype=np.float32), SAMPLE_RATE

        audio_chunks = []
        sample_rate = SAMPLE_RATE

        for i in tqdm(
            range(0, len(chunks), self.config.batch_size),
            desc=desc,
            unit="batch",
            leave=False,
        ):
            batch_texts = chunks[i : i + self.config.batch_size]
            batch_audio, sample_rate = generation_fn(batch_texts)
            if isinstance(batch_audio, np.ndarray):
                batch_audio = [batch_audio]
            audio_chunks.extend(batch_audio)

        return (
            concatenate_audio(audio_chunks, sample_rate, PARAGRAPH_PAUSE_MS),
            sample_rate,
        )

    def synthesize_long(self, text: str, instruct: str = "") -> tuple[np.ndarray, int]:
        """synthesize long text by chunking at sentence boundaries."""
        return self._generate_long(
            text,
            lambda batch: self.synthesize(batch, instruct),
            desc="synthesizing chunks",
        )

    def clone_voice_long(
        self,
        text: str,
        ref_audio: np.ndarray | tuple | str,
        ref_text: str,
    ) -> tuple[np.ndarray, int]:
        """clone voice for long text by chunking at sentence boundaries."""
        return self._generate_long(
            text,
            lambda batch: self.clone_voice(batch, ref_audio, ref_text),
            desc="cloning chunks",
        )


def synthesize_chapters(
    workdir: Path,
    config: Any = None,
    chapters: list[int] | None = None,
    instruct: str = "",
    pooled: bool = False,
    force: bool = False,
    retake: bool = False,
    only_hashes: set[str] | None = None,
) -> None:
    """synthesize audio for chapters in workdir."""
    from .tts_http import HTTPTTSConfig

    if not isinstance(config, HTTPTTSConfig):
        _require_local()
    extract_dir = get_command_dir(workdir, "extract")
    synth_dir = get_command_dir(workdir, "synthesize")

    # check if any source files exist
    if not any(extract_dir.glob(f"*{TXT_EXT}")):
        msg = "synthesize: no text files found in extract/!"
        print(msg)
        raise RuntimeError(msg)

    from .utils import create_tts_engine

    engine = create_tts_engine(config)
    metadata = load_metadata(workdir)
    pending = [
        (s, t)
        for _, s, t in list_chapters(
            metadata, extract_dir, synth_dir, chapters_filter=chapters
        )
    ]

    if not pending:
        msg = "synthesize: no chapters to process."
        print(msg)
        raise RuntimeError(msg)

    # resume manager for assembly
    resume = ResumeManager.for_command(workdir, "synthesize", force=force)

    # pooled and single-chapter now use the same underlying segment-based logic
    _perform_synthesis(
        engine,
        pending,
        instruct,
        resume=resume,
        force=force,
        retake=retake,
        only_hashes=only_hashes,
    )


def _perform_synthesis(
    engine: TTSEngine,
    pending: list[tuple[Path, Path]],
    instruct: str = "",
    resume: ResumeManager | None = None,
    force: bool = False,
    retake: bool = False,
    only_hashes: set[str] | None = None,
) -> None:
    """synthesize multiple chapters with pooled batching and segment caching."""
    chapter_data = []

    # resolve voice reference if provided (for whole book cloning)
    voice_path: Path | None = None
    voice_text: str | None = None

    # check if we should look for a voice
    voice_name = getattr(engine.config, "voice", None)
    if voice_name:
        # look in audition/ for the per-character base voice
        workdir = pending[0][0].parent.parent
        p = get_command_dir(workdir, "audition") / f"{voice_name}{WAV_EXT}"
        if p.exists():
            voice_path = p
            # try to load cast to get text
            from .dramatize import load_cast

            cast = load_cast(workdir)
            char = next((c for c in cast if c.name == voice_name), None)
            if char:
                voice_text = char.audition_line

    for txt_path, wav_path in pending:
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text, engine.config.chunk_size)
        chunks = [c for c in chunks if c.strip()]

        segments_dir = get_segments_dir(wav_path.parent)
        tasks = []
        for chunk in chunks:
            # include voice in hash if present
            hash_data = {"text": chunk, "instruct": instruct}
            if voice_path:
                hash_data["voice_path"] = str(voice_path)

            chunk_hash = compute_hash(hash_data)
            tasks.append(
                AudioTask(
                    text=chunk,
                    segment_hash=chunk_hash,
                    segments_dir=segments_dir,
                    instruct=instruct,
                    voice_ref_audio=voice_path,
                    voice_ref_text=voice_text,
                )
            )
        chapter_data.append((wav_path, tasks))

    process_audio_pipeline(
        engine,
        chapter_data,
        resume=resume,
        desc="synthesizing chapters",
        force=force,
        retake=retake,
        only_hashes=only_hashes,
    )
