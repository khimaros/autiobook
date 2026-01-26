"""showcase command implementation."""

from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf  # type: ignore
from tqdm import tqdm  # type: ignore

from .audio import get_segments_dir
from .config import (
    BASE_MODEL,
    CAST_FILE,
    SHOWCASE_EMOTIONS,
    WAV_EXT,
)
from .dramatize import load_cast
from .llm import Character
from .pooling import AudioTask
from .resume import ResumeManager, compute_hash, get_command_dir
from .tts import TTSConfig, TTSEngine, TXT_EXT


def _showcase_voice(
    workdir: Path,
    cast_list: List[Character] | None,
    voice_name: str,
    text: str | None,
    emotions: list[str] | None,
    force: bool,
) -> None:
    """interactive showcase for a specific voice."""
    engine = TTSEngine(TTSConfig(model_name=BASE_MODEL))
    showcase_dir = get_command_dir(workdir, "showcase")

    # check if cast member
    char = next(
        (c for c in (cast_list or []) if c.name.lower() == voice_name.lower()), None
    )

    # resolve voice path (audition dir)
    voice_path = None
    ref_text = None

    if char:
        audition_dir = get_command_dir(workdir, "audition")
        p = audition_dir / f"{char.name}{WAV_EXT}"
        if p.exists():
            voice_path = p
            ref_text = char.audition_line
    elif voice_name:
        # try to look up in audition dir even if not in cast list (e.g. manually added)
        audition_dir = get_command_dir(workdir, "audition")
        p = audition_dir / f"{voice_name}{WAV_EXT}"
        if p.exists():
            voice_path = p
            # without cast entry, we don't have ref_text, will fallback to dummy

    # determine emotions
    target_emotions = emotions or list(SHOWCASE_EMOTIONS.keys())

    print(f"showcase: generating samples for '{voice_name}'...")

    char_dir = showcase_dir / voice_name
    char_dir.mkdir(parents=True, exist_ok=True)

    for emotion in target_emotions:
        if emotion not in SHOWCASE_EMOTIONS:
            print(f"  unknown emotion '{emotion}', skipping")
            continue

        instruct, sample_line = SHOWCASE_EMOTIONS[emotion]
        # use provided text or default sample line
        input_text = text or sample_line

        wav_path = char_dir / f"{emotion}{WAV_EXT}"

        if not force and wav_path.exists() and not text:
            print(f"  skipping {emotion} (exists)")
            continue

        print(f"  generating {emotion}: '{input_text}'")

        try:
            if voice_path:
                if not ref_text:
                    print(
                        f"    warning: no reference text found for {voice_name}, cloning may vary"
                    )
                    ref_text = (
                        "The quick brown fox jumps over the lazy dog."  # dummy ref text
                    )

                audio, sr = engine.clone_voice(input_text, str(voice_path), ref_text)
            else:
                # base voice mode
                audio, sr = engine.synthesize(
                    input_text, instruct=instruct, speaker=voice_name
                )

            sf.write(str(wav_path), audio, sr)
            print(f"    saved to {wav_path}")

        except Exception as e:
            print(f"    failed: {e}")


def _showcase_pooled(
    engine: TTSEngine,
    chapter_data: list[tuple[Path, list[AudioTask]]],
    resume: ResumeManager,
    verbose: bool = False,
) -> None:
    """generate showcase samples using pooled batching."""
    # flatten all tasks for batch processing
    all_tasks = []
    for _, tasks in chapter_data:
        all_tasks.extend(tasks)

    if not all_tasks:
        return

    # group by voice reference for efficient batching
    voice_groups: dict[str, list[AudioTask]] = {}
    for task in all_tasks:
        key = str(task.voice_ref_audio)
        if key not in voice_groups:
            voice_groups[key] = []
        voice_groups[key].append(task)

    for voice_ref, tasks in tqdm(voice_groups.items(), desc="processing voices"):
        # all tasks here need generation (filtered by caller)
        texts = [t.text for t in tasks]
        ref_audio = tasks[0].voice_ref_audio
        ref_text = tasks[0].voice_ref_text
        assert ref_audio is not None and ref_text is not None

        try:
            audios, sr = engine.clone_voice(texts, str(ref_audio), ref_text)
            if not isinstance(audios, list):
                audios = [audios]

            # save directly to final location
            for task, audio in zip(tasks, audios):
                metadata = task.metadata or {}
                char_name = metadata.get("char_name", "unknown")
                emotion = metadata.get("emotion", "unknown")
                resume_key = metadata.get("resume_key", f"{char_name}/{emotion}")

                # derive output path from segments_dir parent (showcase dir)
                showcase_dir = task.segments_dir.parent
                wav_path = showcase_dir / char_name / f"{emotion}{WAV_EXT}"

                sf.write(str(wav_path), audio, sr)
                resume.update(resume_key, task.segment_hash)

                if verbose:
                    tqdm.write(f"  saved {char_name}/{emotion}")

        except Exception as e:
            print(f"  failed to generate for {voice_ref}: {e}")
            continue

    resume.save()


def run_showcase(
    workdir: Path,
    cast: List[Character] | None = None,
    verbose: bool = False,
    force: bool = False,
    voice: str | None = None,
    text: str | None = None,
    emotions: list[str] | None = None,
) -> None:
    """generate emotion samples for each character voice using cloning."""

    if cast is None:
        cast = load_cast(workdir)

    if voice:
        # also look in voices dir for single voice showcase
        if not cast and not (workdir / "voices" / f"{voice}{WAV_EXT}").exists():
            # if no cast file AND no voice file found, then warn
            if not (get_command_dir(workdir, "cast") / CAST_FILE).exists():
                print("cast file not found. checking for voice file...")

        _showcase_voice(workdir, cast, voice, text, emotions, force)
        return

    if not cast:
        cast_path = get_command_dir(workdir, "cast") / CAST_FILE
        if cast_path.exists():
            print("cast file found but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return

    voices_dir = get_command_dir(workdir, "audition")
    showcase_dir = get_command_dir(workdir, "showcase")
    resume = ResumeManager.for_command(workdir, "showcase", force=force)

    # verify audition samples exist
    missing_voices = []
    for char in cast:
        voice_path = voices_dir / f"{char.name}{WAV_EXT}"
        if not voice_path.exists():
            missing_voices.append(char.name)

    if missing_voices:
        print(f"missing audition samples: {', '.join(missing_voices)}")
        print("run 'audition' command first.")
        return

    # build tasks for pooled synthesis
    engine = TTSEngine(TTSConfig(model_name=BASE_MODEL))
    segments_dir = get_segments_dir(showcase_dir)

    chapter_data = []
    total_tasks = 0
    skipped_count = 0

    for char in cast:
        char_dir = showcase_dir / char.name
        char_dir.mkdir(parents=True, exist_ok=True)

        voice_path = voices_dir / f"{char.name}{WAV_EXT}"
        ref_text = char.audition_line

        # hash character voice for cache key stability
        char_hash = compute_hash(
            {
                "name": char.name,
                "description": char.description,
                "audition_line": char.audition_line,
            }
        )

        tasks = []
        for emotion, (instruct, sample_line) in SHOWCASE_EMOTIONS.items():
            wav_path = char_dir / f"{emotion}{WAV_EXT}"

            # hash includes character + emotion + content for resumability
            task_data = {
                "char_hash": char_hash,
                "emotion": emotion,
                "instruct": instruct,
                "sample_line": sample_line,
            }
            task_hash = compute_hash(task_data)
            resume_key = f"{char.name}/{emotion}"

            if (
                not force
                and wav_path.exists()
                and resume.is_fresh(resume_key, task_hash)
            ):
                skipped_count += 1
                if verbose:
                    print(f"  skipping {resume_key} (up to date)")
                continue

            tasks.append(
                AudioTask(
                    text=sample_line,
                    segment_hash=task_hash,
                    segments_dir=segments_dir,
                    voice_ref_audio=voice_path,
                    voice_ref_text=ref_text,
                    instruct=instruct,
                    metadata={
                        "char_name": char.name,
                        "emotion": emotion,
                        "resume_key": resume_key,
                    },
                )
            )
            total_tasks += 1

        if tasks:
            # each character's emotions are grouped for assembly tracking
            # use a placeholder wav_path that won't be used (we save individually)
            chapter_data.append((char_dir / "_placeholder.wav", tasks))

    if not chapter_data:
        print(f"showcase: all {skipped_count} samples up to date.")
        return

    print(f"showcase: generating {total_tasks} samples for {len(cast)} characters...")

    # process using pooled batching for efficiency
    # we handle saving individually since output structure differs from chapters
    _showcase_pooled(engine, chapter_data, resume, verbose)

    print(f"showcase: {total_tasks} generated, {skipped_count} skipped")


def cmd_showcase(args):
    run_showcase(
        Path(args.workdir),
        verbose=args.verbose,
        force=args.force,
        voice=args.voice,
        text=args.text,
        emotions=args.emotion,
    )
