"""tts engine wrapper for qwen3-tts with rocm optimizations."""

import os
import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import transformers
from tqdm import tqdm

from .audio import (
    check_segment_exists,
    concatenate_audio,
    get_segments_dir,
    load_segment,
)
from .config import (
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    MAX_CHUNK_SIZE,
    PARAGRAPH_PAUSE_MS,
    SAMPLE_RATE,
)
from .pooling import AudioTask, process_pooled_tasks
from .resume import compute_hash
from .utils import iter_pending_chapters

transformers.logging.set_verbosity_error()

SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

# warmup text for model compilation
WARMUP_TEXT = "Hello, this is a warmup."


def get_default_device() -> str:
    """get default device (cuda/rocm if available, else cpu)."""
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
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def setup_rocm_env():
    """set environment variables for optimal rocm performance."""
    if not is_rocm():
        return
    # enable experimental aotriton kernels
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    # enable flash attention on amd
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")
    # pre-allocate MIOpen workspace to avoid fallback to slower kernels
    # 256MB should cover most GEMM operations
    os.environ.setdefault("MIOPEN_WORKSPACE_MAX", "256000000")
    # use immediate mode for faster kernel selection
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    # cache compiled kernels to speed up subsequent runs
    os.environ.setdefault("MIOPEN_USER_DB_PATH", os.path.expanduser("~/.cache/miopen"))


@dataclass
class TTSConfig:
    """configuration for tts generation."""

    model_name: str = DEFAULT_MODEL
    speaker: str = DEFAULT_SPEAKER
    language: str = "English"
    device: str = field(default_factory=get_default_device)
    batch_size: int = 64  # batch 64 shows 7x throughput vs batch 1
    chunk_size: int = MAX_CHUNK_SIZE  # 500 chars balances coherence and speed
    compile_model: bool = True  # use torch.compile for faster inference
    warmup: bool = True  # warmup model on first load
    # generation parameters - can tune for speed vs quality
    do_sample: bool = True  # False = greedy (faster), True = sampling (better quality)
    temperature: float = 0.9  # lower = faster/more deterministic
    max_new_tokens: int = 2048  # limit output length per chunk


class TTSEngine:
    """wrapper for qwen3-tts model with rocm optimizations."""

    def __init__(self, config: TTSConfig | None = None):
        setup_rocm_env()
        self.config = config or TTSConfig()
        self._model = None
        self._loaded_model_name = None  # track loaded model name
        self._compiled = False
        self._sdpa_backends = None  # store backends, create context each time

    def _load_model(self):
        """lazy load and optimize the tts model."""
        if (
            self._model is not None
            and self._loaded_model_name == self.config.model_name
        ):
            return

        # unload existing model if different
        if self._model is not None:
            print(f"unloading model {self._loaded_model_name}...")
            del self._model
            torch.cuda.empty_cache()
            self._model = None
            self._loaded_model_name = None

        from qwen_tts import Qwen3TTSModel

        # determine dtype and attention implementation
        if "cuda" in self.config.device:
            dtype = torch.bfloat16
            attn_impl = "sdpa" if is_rocm() else "flash_attention_2"
        elif "mps" in self.config.device:
            dtype = torch.float32
            attn_impl = "sdpa"
        else:
            dtype = torch.float32
            attn_impl = "sdpa"

        print(
            f"loading model {self.config.model_name} on {self.config.device} "
            f"with {dtype} and {attn_impl}..."
        )
        self._model = Qwen3TTSModel.from_pretrained(
            self.config.model_name,
            device_map=self.config.device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        self._loaded_model_name = self.config.model_name

        # setup attention backends for rocm (context created fresh each call)
        if "cuda" in self.config.device and is_rocm():
            try:
                from torch.nn.attention import SDPBackend

                self._sdpa_backends = [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                ]
            except ImportError:
                self._sdpa_backends = None
        else:
            self._sdpa_backends = None

        # apply torch.compile for faster inference
        if self.config.compile_model and "cuda" in self.config.device:
            self._compile_model()

        # warmup the model
        if self.config.warmup:
            self._warmup()

    def _get_attn_ctx(self):
        """create fresh attention context for each use."""
        if self._sdpa_backends is not None:
            from torch.nn.attention import sdpa_kernel

            return sdpa_kernel(self._sdpa_backends)
        return nullcontext()

    def _compile_model(self):
        """apply torch.compile to model components."""
        if self._compiled:
            return

        print("compiling model for faster inference...")
        try:
            # compile the main talker model with reduce-overhead mode
            # this optimizes for inference with minimal CPU overhead
            self._model.model.talker = torch.compile(
                self._model.model.talker,
                mode="reduce-overhead",
                fullgraph=False,  # allow graph breaks for compatibility
            )
            self._compiled = True
            print("model compilation complete")
        except Exception as e:
            print(f"warning: torch.compile failed ({e}), using eager mode")
            self._compiled = False

    def _warmup(self):
        """warmup model with a short synthesis to trigger compilation."""
        print("warming up model...")
        try:
            if "VoiceDesign" in self.config.model_name:
                self.design_voice(WARMUP_TEXT, "neutral voice")
            elif "Base" in self.config.model_name:
                pass
            else:
                self.synthesize(WARMUP_TEXT)
            print("warmup complete")
        except Exception as e:
            print(f"warning: warmup failed ({e})")

    def _run_inference(self, func_name: str, **kwargs) -> tuple[Any, int]:
        """helper to run inference with correct context and params."""
        self._load_model()
        func = getattr(self._model, func_name)

        # common args
        kwargs.setdefault("language", self.config.language)
        kwargs.setdefault("non_streaming_mode", True)
        kwargs.setdefault("do_sample", self.config.do_sample)
        kwargs.setdefault("temperature", self.config.temperature)

        with self._get_attn_ctx():
            with torch.inference_mode():
                return func(**kwargs)

    def synthesize(
        self, text: str | list[str], instruct: str = ""
    ) -> tuple[np.ndarray | list[np.ndarray], int]:
        """synthesize speech from text using current model."""
        wavs, sr = self._run_inference(
            "generate_custom_voice",
            text=text,
            speaker=self.config.speaker,
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


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE) -> list[str]:
    """split text into chunks at sentence boundaries."""
    sentences = SENTENCE_ENDINGS.split(text)

    chunks = []
    current_chunk = []
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

        current_chunk.append(sentence)
        current_length += sentence_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def synthesize_chapters(
    workdir: Path,
    config: TTSConfig | None = None,
    chapters: list[int] | None = None,
    instruct: str = "",
    pooled: bool = False,
    force: bool = False,
) -> None:
    """synthesize audio for chapters in workdir."""
    from .config import TXT_EXT

    # check if any source files exist
    if not any(workdir.glob(f"*{TXT_EXT}")):
        print("synthesize: no text files found in workdir!")
        return

    engine = TTSEngine(config)
    pending = list(
        iter_pending_chapters(
            workdir, chapters, skip_message="already synthesized", force=force
        )
    )

    if not pending:
        print("synthesize: all chapters up to date.")
        return

    # pooled and single-chapter now use the same underlying segment-based logic
    _perform_synthesis(engine, pending, instruct, force=force)


def _perform_synthesis(
    engine: TTSEngine,
    pending: list[tuple[Path, Path]],
    instruct: str = "",
    force: bool = False,
) -> None:
    """synthesize multiple chapters with pooled batching and segment caching."""
    tasks = []
    chapter_sequences = []

    for txt_path, wav_path in pending:
        text = txt_path.read_text()
        chunks = chunk_text(text, engine.config.chunk_size)
        chunks = [c for c in chunks if c.strip()]

        segments_dir = get_segments_dir(wav_path.parent)

        chapter_hashes = []
        for chunk in chunks:
            chunk_hash = compute_hash({"text": chunk, "instruct": instruct})
            chapter_hashes.append(chunk_hash)

            if force or not check_segment_exists(segments_dir, chunk_hash):
                tasks.append(
                    AudioTask(
                        text=chunk,
                        segment_hash=chunk_hash,
                        segments_dir=segments_dir,
                        instruct=instruct,
                    )
                )

        chapter_sequences.append((wav_path, chapter_hashes, segments_dir))

    if tasks:
        process_pooled_tasks(engine, tasks, desc="synthesizing chapters", force=force)

    # Assemble all chapters
    for wav_path, hashes, seg_dir in chapter_sequences:
        try:
            audio_segments = [load_segment(seg_dir, h) for h in hashes]
            combined = concatenate_audio(
                audio_segments, SAMPLE_RATE, PARAGRAPH_PAUSE_MS
            )
            sf.write(str(wav_path), combined, SAMPLE_RATE)
            print(f"  -> {wav_path.name}")
        except Exception as e:
            print(f"failed to assemble {wav_path.name}: {e}")
