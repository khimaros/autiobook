"""shared pooling logic for batch processing."""

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf  # type: ignore
from tqdm import tqdm  # type: ignore

from .audio import (
    check_segment_exists,
    concatenate_audio,
    get_segment_path,
    load_segment,
    save_segment,
)
from .config import SAMPLE_RATE
from .resume import ResumeManager, compute_hash

SEGMENT_STATE_PREFIX = "segment:"


def _segment_wav_sha(
    segments_dir: Path, h: str, resume: ResumeManager | None
) -> str | None:
    """get wav-bytes sha for a segment, preferring persisted state."""
    if resume is not None:
        entry = resume.state.get(f"{SEGMENT_STATE_PREFIX}{h}")
        if isinstance(entry, dict):
            sha = entry.get("wav_sha256")
            if sha:
                return str(sha)
    p = get_segment_path(segments_dir, h)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _chapter_fingerprint(
    segments_dir: Path,
    hashes: list[str],
    resume: ResumeManager | None = None,
) -> str:
    """fingerprint a chapter by its chunk identities *and* chunk wav contents.

    detects regenerated segment wavs even when the logical chunk hash is the
    same (e.g. re-synthesis for a tts fix). falls back to a hash of chunk
    identities when any chunk wav is missing (first-time build), so the
    fingerprint is still stable enough to drive is_fresh checks."""
    parts: list[tuple[str, str]] = []
    for h in hashes:
        sha = _segment_wav_sha(segments_dir, h, resume)
        if sha is None:
            return compute_hash(hashes)
        parts.append((h, sha))
    return compute_hash(parts)


@dataclass
class AudioTask:
    """represents a single unit of audio generation work."""

    text: str
    segment_hash: str
    segments_dir: Path
    voice_ref_audio: Optional[Path] = None
    voice_ref_text: Optional[str] = None
    instruct: str = ""
    # preset-voice mode: use backend voice_id + instruct (no clone, no ref audio)
    preset_voice: Optional[str] = None
    chapter_idx: int = 0  # for chapter-ordered scheduling
    metadata: Optional[dict[str, Any]] = None  # arbitrary metadata for callbacks


@dataclass
class ChapterState:
    """tracks synthesis and assembly state for a chapter."""

    wav_path: Path
    hashes: list[str]
    segments_dir: Path
    manifest_hash: str
    pending_hashes: set[str] = field(default_factory=set)
    assembled: bool = False
    # per-chunk metadata aligned with `hashes` (script_idx, chunk_idx, script_path, ...)
    chunk_meta: list[dict[str, Any]] | None = None


def _log_verbose_tasks(batch: list[AudioTask]) -> None:
    """print voice and text for each task in the batch."""
    for t in batch:
        if t.preset_voice:
            voice = f"preset:{t.preset_voice}"
        elif t.voice_ref_audio:
            voice = t.voice_ref_audio.stem
        else:
            voice = "(default)"
        text = t.text.strip()
        tqdm.write(f"perform: [{voice}] {text}")


def _run_synthesis(engine: Any, batch: list[AudioTask]) -> list[np.ndarray]:
    """invoke the engine on a batch and return aligned audio arrays.

    handles batch padding (for torch.compile) and OOM retry; does not save."""
    texts = [t.text for t in batch]
    ref_audio = batch[0].voice_ref_audio
    ref_text = batch[0].voice_ref_text
    instruct = batch[0].instruct
    preset_voice = batch[0].preset_voice

    original_size = len(texts)
    padded = False
    if (
        hasattr(engine, "config")
        and getattr(engine.config, "compile_model", False)
        and len(texts) < engine.config.batch_size
    ):
        pad_size = engine.config.batch_size - len(texts)
        texts.extend(["."] * pad_size)
        padded = True

    def _call():
        if preset_voice:
            w, _ = engine.synthesize(texts, instruct, speaker=preset_voice)
        elif ref_audio:
            w, _ = engine.clone_voice(texts, ref_audio, ref_text)
        else:
            w, _ = engine.synthesize(texts, instruct)
        return w

    try:
        wavs = _call()
    except RuntimeError as e:
        if "out of memory" not in str(e):
            print(f"batch generation failed: {e}")
            raise
        print("warning: OOM detected, clearing cache and retrying...")
        import torch

        torch.cuda.empty_cache()
        wavs = _call()

    if isinstance(wavs, np.ndarray):
        wavs = [wavs]
    if padded:
        wavs = wavs[:original_size]
    return list(wavs)


MAX_RETAKE_ATTEMPTS = 5


class RetakeError(RuntimeError):
    """raised when retake retries exhaust and segments still fail heuristics."""


def _retry_bad_takes(
    engine: Any,
    batch: list[AudioTask],
    wavs: list[np.ndarray],
    max_attempts: int = MAX_RETAKE_ATTEMPTS,
    verbose: bool = False,
) -> list[np.ndarray]:
    """re-synthesize any wavs flagged by retake heuristics; bumps seed each try.

    raises RetakeError if any segment still fails after max_attempts."""
    from .retake import categorize_audio

    base_seed = int(getattr(engine.config, "seed", 0) or 0)
    original_seed = base_seed

    had_retakes = False
    for attempt in range(1, max_attempts + 1):
        flagged = [(i, categorize_audio(a)) for i, a in enumerate(wavs)]
        bad = [(i, cats) for i, cats in flagged if cats]
        if not bad:
            if had_retakes and verbose:
                tqdm.write(f"retake: attempt {attempt}/{max_attempts}: passed")
            break
        had_retakes = True

        # fresh random seed per retry explores a genuinely different trajectory;
        # neighboring seeds (base+1, base+2) often produce similar failures.
        new_seed = random.randint(1, 2**31 - 1) if base_seed > 0 else 0

        if verbose:
            for i, cats in bad:
                tqdm.write(
                    f"retake: attempt {attempt}/{max_attempts}: "
                    f"{batch[i].segment_hash[:16]} ({','.join(cats)}) "
                    f"seed={new_seed}; retrying..."
                )

        if base_seed > 0:
            engine.config.seed = new_seed

        sub_idx = [i for i, _ in bad]
        sub = [batch[i] for i in sub_idx]
        try:
            redo = _run_synthesis(engine, sub)
        except Exception as e:
            tqdm.write(f"retake: retry {attempt} failed: {e}")
            break
        for k, i in enumerate(sub_idx):
            wavs[i] = redo[k]

    if base_seed > 0:
        engine.config.seed = original_seed

    final_bad = [
        (batch[i], categorize_audio(a))
        for i, a in enumerate(wavs)
        if categorize_audio(a)
    ]
    if final_bad:
        parts = [
            f"{task.segment_hash[:16]} ({','.join(cats)})" for task, cats in final_bad
        ]
        raise RetakeError(
            f"retake: gave up on {len(final_bad)} segment(s) after "
            f"{max_attempts} attempt(s): {'; '.join(parts)}"
        )
    return wavs


def _synthesize_batch(
    engine: Any,
    batch: list[AudioTask],
    pbar: tqdm | None = None,
    verbose: bool = False,
    retake: bool = False,
    resume: ResumeManager | None = None,
) -> None:
    """synthesize a batch of tasks with the same voice reference."""
    if not batch:
        return

    if verbose:
        _log_verbose_tasks(batch)

    try:
        wavs = _run_synthesis(engine, batch)
    except Exception as e:
        print(f"batch generation failed: {e}")
        raise

    if retake:
        wavs = _retry_bad_takes(engine, batch, wavs, verbose=verbose)

    seed = int(getattr(engine.config, "seed", 0) or 0)
    for j, audio in enumerate(wavs):
        if j < len(batch):
            wav_sha = save_segment(
                batch[j].segments_dir,
                batch[j].segment_hash,
                audio,
                SAMPLE_RATE,
            )
            if resume is not None:
                resume.update(
                    f"{SEGMENT_STATE_PREFIX}{batch[j].segment_hash}",
                    batch[j].segment_hash,
                    wav_sha256=wav_sha,
                    seed=seed,
                )
        if pbar:
            pbar.update(1)
    if resume is not None:
        resume.save()


PAUSE_MS_BETWEEN_CHUNKS = 500


def timing_manifest_path(wav_path: Path) -> Path:
    """path to the chunk-timing manifest for a chapter wav."""
    return wav_path.with_suffix(wav_path.suffix + ".timing.json")


def srt_path(wav_path: Path) -> Path:
    """path to the .srt subtitles file for a chapter wav."""
    return wav_path.with_suffix(".srt")


def vtt_path(wav_path: Path) -> Path:
    """path to the .vtt subtitles file for a chapter wav."""
    return wav_path.with_suffix(".vtt")


def _format_ts(seconds: float, sep: str) -> str:
    """format seconds as HH:MM:SS<sep>mmm (sep=',' for srt, '.' for vtt)."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60000) % 60
    h = total_ms // 3600000
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _iter_cues(chunks: list[dict[str, Any]]):
    """yield (start, end, body) for each non-empty chunk with speaker prefix."""
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        speaker = c.get("speaker")
        body = f"[{speaker}] {text}" if speaker else text
        yield c["start_s"], c["end_s"], body


def write_subtitles(wav_path: Path, chunks: list[dict[str, Any]]) -> None:
    """write SRT and VTT files next to the chapter wav."""
    srt_lines: list[str] = []
    vtt_lines: list[str] = ["WEBVTT", ""]
    for idx, (start, end, body) in enumerate(_iter_cues(chunks), start=1):
        srt_lines.extend(
            [
                str(idx),
                f"{_format_ts(start, ',')} --> {_format_ts(end, ',')}",
                body,
                "",
            ]
        )
        vtt_lines.extend(
            [
                f"{_format_ts(start, '.')} --> {_format_ts(end, '.')}",
                body,
                "",
            ]
        )
    srt_path(wav_path).write_text("\n".join(srt_lines), encoding="utf-8")
    vtt_path(wav_path).write_text("\n".join(vtt_lines), encoding="utf-8")


def write_timing_manifest(
    wav_path: Path,
    segments_dir: Path,
    hashes: list[str],
    chunk_meta: list[dict[str, Any]] | None,
    sample_rate: int = SAMPLE_RATE,
    pause_ms: int = PAUSE_MS_BETWEEN_CHUNKS,
) -> None:
    """write a timing manifest (JSON) + subtitles (SRT) for a chapter wav."""
    import json

    pause_s = pause_ms / 1000.0
    chunks: list[dict[str, Any]] = []
    offset_s = 0.0
    for i, h in enumerate(hashes):
        info = sf.info(str(segments_dir / f"{h}.wav"))
        dur_s = info.frames / info.samplerate
        entry: dict[str, Any] = {
            "hash": h,
            "start_s": offset_s,
            "end_s": offset_s + dur_s,
        }
        if chunk_meta and i < len(chunk_meta):
            entry.update(chunk_meta[i])
        chunks.append(entry)
        offset_s += dur_s
        if i < len(hashes) - 1:
            offset_s += pause_s

    manifest = {
        "version": 1,
        "sample_rate": sample_rate,
        "pause_ms": pause_ms,
        "chunks": chunks,
    }
    timing_manifest_path(wav_path).write_text(json.dumps(manifest, indent=2))
    write_subtitles(wav_path, chunks)


def _assemble_chapter(
    state: ChapterState,
    resume: ResumeManager | None = None,
) -> bool:
    """assemble a chapter from its segments. returns True on success."""
    print(f"assembling {state.wav_path.name}...")
    try:
        audio = concatenate_audio(
            [load_segment(state.segments_dir, h) for h in state.hashes]
        )
        sf.write(str(state.wav_path), audio, SAMPLE_RATE)
        write_timing_manifest(
            state.wav_path, state.segments_dir, state.hashes, state.chunk_meta
        )
        if resume:
            # recompute post-synthesis: fingerprint at state-build time may
            # predate any chunks that were generated during this run.
            final_hash = _chapter_fingerprint(state.segments_dir, state.hashes, resume)
            resume.update(str(state.wav_path), final_hash)
            resume.save()
        print(f"  -> {state.wav_path.name}")
        state.assembled = True
        return True
    except Exception as e:
        print(f"failed to assemble {state.wav_path.name}: {e}")
        raise


def _try_assemble_ready_chapters(
    chapters: list[ChapterState],
    generated_hashes: set[str],
    resume: ResumeManager | None = None,
) -> None:
    """assemble chapters that have all segments ready, in order."""
    for ch in chapters:
        if ch.assembled:
            continue
        # update pending hashes based on what's been generated
        ch.pending_hashes -= generated_hashes
        # can only assemble if all segments are ready
        if ch.pending_hashes:
            # stop at first incomplete chapter to maintain order
            break
        _assemble_chapter(ch, resume)


def _get_voice_key(task: AudioTask) -> tuple:
    """get the voice grouping key for a task."""
    if task.preset_voice:
        return ("preset", task.preset_voice, task.instruct)
    return (
        str(task.voice_ref_audio) if task.voice_ref_audio else None,
        task.voice_ref_text,
    )


def _select_batch_for_chapter(
    chapter_idx: int,
    tasks_by_voice: dict[tuple, list[AudioTask]],
    batch_size: int,
    needed_hashes: set[str],
) -> list[AudioTask]:
    """select a batch of tasks prioritizing the given chapter.

    for voice cloning, we batch by voice but prefer tasks needed by the current chapter.
    this increases the chance of completing earlier chapters sooner.
    """
    # find voice groups that have tasks for this chapter
    for voice_key, tasks in tasks_by_voice.items():
        # get tasks needed by this chapter first
        batch = []
        for t in tasks:
            if t.segment_hash in needed_hashes:
                batch.append(t)
                if len(batch) == batch_size:
                    break

        if batch:
            # fill remaining slots with tasks from same voice, any chapter
            if len(batch) < batch_size:
                for t in tasks:
                    if t not in batch:
                        batch.append(t)
                        if len(batch) == batch_size:
                            break

            # remove selected tasks from the pool
            for t in batch:
                tasks.remove(t)
            # clean up empty voice groups
            if not tasks:
                del tasks_by_voice[voice_key]
            return batch
    return []


def process_audio_pipeline(
    engine: Any,
    chapter_data: list[tuple[Path, list[AudioTask]]],
    resume: ResumeManager | None = None,
    desc: str = "generating audio",
    force: bool = False,
    verbose: bool = False,
    retake: bool = False,
    only_hashes: set[str] | None = None,
) -> None:
    """unified pipeline for batch processing audio segments and assembling chapters.

    processes chapters in order, assembling each as soon as all its segments are ready.
    for voice cloning, batches by voice but prioritizes earlier chapters.
    """
    if not chapter_data:
        return

    # build chapter states and collect all pending tasks
    chapters: list[ChapterState] = []
    all_pending_tasks: list[AudioTask] = []
    generated_hashes: set[str] = set()  # track globally generated segments

    for ch_idx, (wav_path, tasks) in enumerate(chapter_data):
        if not tasks:
            continue

        seg_dir = tasks[0].segments_dir
        hashes = [t.segment_hash for t in tasks]
        m_hash = _chapter_fingerprint(seg_dir, hashes, resume)
        # always capture text (and any caller-provided metadata like speaker)
        # so the timing manifest and subtitles can be written.
        chunk_meta = []
        for t in tasks:
            entry: dict[str, Any] = {"text": t.text}
            if t.metadata:
                entry.update(t.metadata)
            chunk_meta.append(entry)

        # determine which segments need generation
        pending_hashes = set()
        for t in tasks:
            t.chapter_idx = ch_idx
            if only_hashes is not None and t.segment_hash not in only_hashes:
                continue
            if force or not check_segment_exists(t.segments_dir, t.segment_hash):
                pending_hashes.add(t.segment_hash)
                # avoid duplicate synthesis across chapters
                if t.segment_hash not in generated_hashes:
                    all_pending_tasks.append(t)
                    generated_hashes.add(t.segment_hash)

        needs_assembly = (
            force
            or not wav_path.exists()
            or (resume and not resume.is_fresh(str(wav_path), m_hash))
        )

        chapters.append(
            ChapterState(
                wav_path=wav_path,
                hashes=hashes,
                segments_dir=seg_dir,
                manifest_hash=m_hash,
                pending_hashes=pending_hashes,
                assembled=not needs_assembly,
                chunk_meta=chunk_meta,
            )
        )

        # backfill missing timing manifest for already-assembled chapters.
        # manifest depends on chunk wavs + meta, not the chapter wav, so we
        # can write it without re-concatenating.
        if (
            not needs_assembly
            and not timing_manifest_path(wav_path).exists()
            and all(check_segment_exists(seg_dir, h) for h in hashes)
        ):
            write_timing_manifest(wav_path, seg_dir, hashes, chunk_meta)

    # reset for tracking during synthesis
    generated_hashes = set()

    if not all_pending_tasks:
        # all segments cached, just assemble
        _try_assemble_ready_chapters(chapters, set(), resume)
        return

    # group tasks by voice for efficient batching
    tasks_by_voice: dict[tuple, list[AudioTask]] = defaultdict(list)
    for task in all_pending_tasks:
        key = _get_voice_key(task)
        tasks_by_voice[key].append(task)

    # sort tasks within each voice group by chapter index (earlier chapters first)
    for tasks in tasks_by_voice.values():
        tasks.sort(key=lambda t: t.chapter_idx)

    total_tasks = len(all_pending_tasks)
    with tqdm(total=total_tasks, desc=desc, unit="seg") as pbar:
        # process chapter by chapter, but batch by voice
        for ch_idx, ch in enumerate(chapters):
            if ch.assembled:
                continue

            # synthesize batches until this chapter is ready
            while ch.pending_hashes - generated_hashes:
                needed = ch.pending_hashes - generated_hashes
                batch = _select_batch_for_chapter(
                    ch_idx, tasks_by_voice, engine.config.batch_size, needed
                )
                if not batch:
                    # no more tasks for any voice, shouldn't happen
                    break

                _synthesize_batch(
                    engine, batch, pbar, verbose=verbose, retake=retake, resume=resume
                )
                for t in batch:
                    generated_hashes.add(t.segment_hash)

                # try to assemble any chapters that are now ready
                _try_assemble_ready_chapters(chapters, generated_hashes, resume)

        # process any remaining tasks (for later chapters)
        while any(tasks_by_voice.values()):
            # pick first non-empty voice group
            for voice_key, tasks in list(tasks_by_voice.items()):
                if tasks:
                    batch = tasks[: engine.config.batch_size]
                    for t in batch:
                        tasks.remove(t)
                    if not tasks:
                        del tasks_by_voice[voice_key]
                    _synthesize_batch(
                        engine,
                        batch,
                        pbar,
                        verbose=verbose,
                        retake=retake,
                        resume=resume,
                    )
                    for t in batch:
                        generated_hashes.add(t.segment_hash)
                    _try_assemble_ready_chapters(chapters, generated_hashes, resume)
                    break

    # final assembly pass for any remaining chapters
    _try_assemble_ready_chapters(chapters, generated_hashes, resume)
