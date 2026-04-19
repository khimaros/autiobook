"dramatization workflow logic."

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, cast

import soundfile as sf  # type: ignore
from tqdm import tqdm  # type: ignore

from .audio import (
    get_segments_dir,
)
from .config import (
    BASE_MODEL,
    CAST_BATCH_SIZE,
    CAST_FILE,
    DEFAULT_CAST,
    DEFAULT_LLM_MODEL,
    DEFAULT_THINKING_BUDGET,
    EMOTION_SEP,
    RETAINED_SPEAKERS,
    SCRIPT_EXT,
    TXT_EXT,
    VALIDATION_MAX_RETRIES,
    VOICE_DESIGN_MODEL,
    VOICE_EMOTIONS,
    WAV_EXT,
)
from .epub import load_metadata
from .llm import (
    Character,
    ScriptSegment,
    fix_missing_segment,
    generate_cast,
    process_script_chunk,
    split_text_smart,
)
from .pooling import AudioTask, process_audio_pipeline
from .resume import ResumeManager, compute_hash, get_command_dir, list_chapters
from .utils import chunk_text, create_tts_engine, dir_mtime, get_chapters


class ValidationError(RuntimeError):
    """validation failed for script generation."""


def save_cast(workdir: Path, cast: List[Character]) -> None:
    """save cast to json file."""

    path = get_command_dir(workdir, "cast") / CAST_FILE

    characters = []
    for c in cast:
        char_data = {
            "name": c.name,
            "description": c.description,
            "audition_line": c.audition_line,
            "aliases": c.aliases,
        }
        characters.append(char_data)

    data = {
        "version": 4,
        "characters": characters,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cast(workdir: Path) -> List[Character]:
    """load cast from json file."""

    path = get_command_dir(workdir, "cast") / CAST_FILE
    if not path.exists():
        return [
            Character(
                name=c["name"],
                description=c["description"],
                audition_line=c["audition_line"],
            )
            for c in DEFAULT_CAST
        ]

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # handle legacy list format
    if isinstance(data, list):
        chars_legacy = []
        for c in data:
            chars_legacy.append(
                Character(
                    name=c["name"],
                    description=c["description"],
                    audition_line=c["audition_line"],
                    aliases=c.get("aliases"),
                )
            )
        return chars_legacy

    # handle dict format
    chars_dict = []
    for c in cast(dict, data).get("characters", []):
        chars_dict.append(
            Character(
                name=c["name"],
                description=c["description"],
                audition_line=c["audition_line"],
                aliases=c.get("aliases"),
            )
        )
    return chars_dict


def save_script(
    script_path: Path,
    segments: List[ScriptSegment],
) -> None:
    """save dramatized script for a chapter."""
    data = {
        "version": 2,
        "segments": [
            {
                "speaker": s.speaker,
                "text": s.text,
                "instruction": s.instruction,
            }
            for s in segments
        ],
    }
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_script(script_path: Path) -> List[ScriptSegment]:
    """load dramatized script for a chapter."""
    if not script_path.exists():
        return []

    with open(script_path, encoding="utf-8") as f:
        data = json.load(f)

    return [ScriptSegment(**s) for s in cast(dict, data).get("segments", [])]


def _find_existing_character(
    c: Character, cast_map: dict[str, Character], alias_map: dict[str, str]
) -> tuple[Optional[Character], Optional[str]]:
    """find an existing character that matches the given one."""
    key = c.name.lower()

    # 1. name is an alias
    if key in alias_map:
        return cast_map[alias_map[key]], c.name

    # 2. any of character's aliases match existing
    if c.aliases:
        for alias in c.aliases:
            a_low = alias.lower()
            if a_low in cast_map:
                return cast_map[a_low], c.name
            if a_low in alias_map:
                return cast_map[alias_map[a_low]], c.name

    # 3. exact match
    return cast_map.get(key), None


def _merge_character_into_cast(
    c: Character,
    cast_map: dict[str, Character],
    alias_map: dict[str, str],
    verbose: bool = False,
) -> str:
    """merge a character into the cast, returns 'added', 'updated', or 'merged'."""
    existing, merge_source = _find_existing_character(c, cast_map, alias_map)

    if existing:
        diff_parts: list[str] = []
        # exclude canonical name from alias comparison — the LLM sometimes emits
        # the canonical name as its own alias, which is cleanup noise, not a diff.
        canon_low = existing.name.casefold()
        old_aliases = {a for a in (existing.aliases or []) if a.casefold() != canon_low}
        new_aliases = set(old_aliases)
        if c.aliases:
            new_aliases.update(a for a in c.aliases if a.casefold() != canon_low)
        if merge_source and merge_source.casefold() != canon_low:
            new_aliases.add(merge_source)

        added_aliases = sorted(new_aliases - old_aliases)
        if new_aliases != set(existing.aliases or []):
            existing.aliases = sorted(new_aliases) if new_aliases else None
        if added_aliases:
            diff_parts.append("+aliases: " + ", ".join(repr(a) for a in added_aliases))

        if c.description and c.description != existing.description:
            diff_parts.append(
                f"description: {existing.description!r} -> {c.description!r}"
            )
            existing.description = c.description

        if verbose and diff_parts:
            label = "merged" if merge_source else "updated"
            print(f"  {label} '{existing.name}':")
            for part in diff_parts:
                print(f"    {part}")

        return "merged" if merge_source else ("updated" if diff_parts else "unchanged")

    # new character: drop any aliases that duplicate the canonical name
    canon_low = c.name.casefold()
    clean_aliases = (
        [a for a in c.aliases if a.casefold() != canon_low] if c.aliases else []
    )
    c.aliases = clean_aliases or None
    if verbose:
        alias_str = (
            f" (aliases: {', '.join(repr(a) for a in clean_aliases)})"
            if clean_aliases
            else ""
        )
        print(f"  added new character: '{c.name}'{alias_str}")
    cast_map[c.name.lower()] = c
    for alias in clean_aliases:
        alias_map[alias.lower()] = c.name.lower()
    return "added"


def _get_chapters_to_analyze(
    chapter_map: dict[int, Path],
    chapters: list[int] | None,
    resume: ResumeManager,
    force: bool,
) -> tuple[list[int], dict[int, str]]:
    """identify which chapters need analysis and compute their hashes."""
    chapters_to_process = []
    chapter_hashes = {}
    candidate_chapters = chapters if chapters else sorted(chapter_map.keys())

    for num in candidate_chapters:
        if num not in chapter_map:
            continue
        txt_path = chapter_map[num]
        text = txt_path.read_text(encoding="utf-8")
        text_hash = compute_hash(text)
        chapter_hashes[num] = text_hash
        if force or not resume.is_fresh(str(num), text_hash):
            chapters_to_process.append(num)

    return chapters_to_process, chapter_hashes


def _process_cast_batch(
    batch_chapters: list[int],
    chapter_map: dict[int, Path],
    cast_map: dict[str, Character],
    alias_map: dict[str, str],
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    verbose: bool,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> int:
    """process a single batch of chapters for cast generation."""
    full_sample = ""
    for num in batch_chapters:
        txt_path = chapter_map[num]
        full_sample += f"\n--- Chapter {txt_path.stem} ---\n"
        full_sample += txt_path.read_text(encoding="utf-8")

    # format current cast for context
    current_cast = list(cast_map.values())
    summary = "\n".join(
        f"- {c.name}: {c.description}"
        + (f" (also known as: {', '.join(c.aliases)})" if c.aliases else "")
        for c in current_cast
    )

    batch_cast = generate_cast(
        full_sample,
        api_base,
        api_key,
        model or DEFAULT_LLM_MODEL,
        existing_cast_summary=summary,
        thinking_budget=thinking_budget,
    )

    added, updated, merged = 0, 0, 0
    for c in batch_cast:
        result = _merge_character_into_cast(c, cast_map, alias_map, verbose=verbose)
        if result == "added":
            added += 1
        elif result == "updated":
            updated += 1
        elif result == "merged":
            merged += 1

    return len(batch_cast)


def run_cast_generation(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    verbose: bool = False,
    force: bool = False,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> List[Character]:
    """analyze book and generate cast list."""
    existing_cast = load_cast(workdir)
    resume = ResumeManager.for_command(workdir, "cast", force=force)

    extract_dir = get_command_dir(workdir, "extract")
    txt_files = sorted(extract_dir.glob(f"*{TXT_EXT}"))
    if not txt_files:
        print("no extracted text files found!")
        return existing_cast

    chapter_map = {}
    for txt_path in txt_files:
        try:
            num = int(txt_path.stem.split("_")[0])
            chapter_map[num] = txt_path
        except ValueError:
            continue

    chapters_to_process, chapter_hashes = _get_chapters_to_analyze(
        chapter_map, chapters, resume, force
    )
    if not chapters_to_process:
        print(f"cast: all {len(chapters or chapter_map)} chapters up to date.")
        return existing_cast

    print(f"cast: analyzing {len(chapters_to_process)} chapters...")

    cast_map = {c.name.lower(): c for c in existing_cast}
    alias_map = {
        a.lower(): c.name.lower() for c in existing_cast if c.aliases for a in c.aliases
    }

    batch_size = CAST_BATCH_SIZE
    for batch_start in range(0, len(chapters_to_process), batch_size):
        batch_chapters = chapters_to_process[batch_start : batch_start + batch_size]
        print(
            f"  batch {batch_start // batch_size + 1}: chapters {batch_chapters} "
            f"({len(cast_map)} characters known)..."
        )

        _process_cast_batch(
            batch_chapters,
            chapter_map,
            cast_map,
            alias_map,
            api_base,
            api_key,
            model,
            verbose,
            thinking_budget,
        )

        for num in batch_chapters:
            resume.update(str(num), chapter_hashes[num])
        resume.save()

        final_cast = list(cast_map.values())
        narrator = next((c for c in final_cast if c.name.lower() == "narrator"), None)
        if narrator:
            final_cast.remove(narrator)
            final_cast.insert(0, narrator)
        save_cast(workdir, final_cast)

    return list(cast_map.values())


def _audition_tasks(
    char: Character,
    audition_line: str | None,
) -> list[tuple[str, str, str, str]]:
    """build (filename_base, resume_key, text, instruct) for each emotion variant."""
    tasks = []
    for emotion, (emotion_instruct, sample_line) in VOICE_EMOTIONS.items():
        filename = f"{char.name}{EMOTION_SEP}{emotion}"
        resume_key = f"{char.name}/{emotion}"
        instruct = f"{char.description}; {emotion_instruct}"
        text = audition_line or sample_line
        tasks.append((filename, resume_key, text, instruct))
    return tasks


def run_auditions(
    workdir: Path,
    cast: List[Character] | None = None,
    verbose: bool = False,
    force: bool = False,
    audition_line: str | None = None,
    config: Any = None,
    callback: bool = False,
) -> None:
    """generate voice samples for cast with emotion variants."""

    if cast is None:
        cast = load_cast(workdir)

    voices_dir = get_command_dir(workdir, "audition")
    resume = ResumeManager.for_command(workdir, "audition", force=force)

    from .introduce import recorded_seed

    if not cast:
        cast_path = get_command_dir(workdir, "cast") / CAST_FILE
        if cast_path.exists():
            print(f"cast file found at {cast_path} but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return

    if len(cast) <= 3 and cast[0].name == "Narrator":
        print(
            "warning: using default cast (Narrator + Extras). "
            "run 'cast' to generate full cast."
        )

    if config is None:
        from .tts import TTSConfig

        config = TTSConfig(model_name=VOICE_DESIGN_MODEL)
    engine = create_tts_engine(config)

    print(
        f"generating auditions for {len(cast)} characters "
        f"({len(VOICE_EMOTIONS)} emotions each)..."
    )

    generated_count = 0
    skipped_count = 0

    for char in tqdm(cast, desc="casting voices"):
        tasks = _audition_tasks(char, audition_line)

        # reuse the per-character seed recorded by introduce so all of a
        # character's ref clips (base + emotion variants) ride the same
        # trajectory. a changed introduce seed forces re-audition via the hash.
        intro_seed = recorded_seed(workdir, char.name)
        if intro_seed > 0:
            engine.config.seed = intro_seed

        for filename, resume_key, text, instruct in tasks:
            wav_path = voices_dir / f"{filename}{WAV_EXT}"

            task_data = {
                "name": char.name,
                "description": char.description,
                "text": text,
                "instruct": instruct,
                "introduce_seed": intro_seed,
            }
            task_hash = compute_hash(task_data)

            if (
                not force
                and wav_path.exists()
                and resume.is_fresh(resume_key, task_hash)
            ):
                skipped_count += 1
                continue

            if verbose:
                tqdm.write(f"  {resume_key}: '{text[:60]}'")

            try:
                if callback:
                    from .callback import generate_with_callback
                    from .retake import get_reject_dir

                    character, _, emotion = resume_key.partition("/")
                    audio, sr = generate_with_callback(
                        lambda: engine.design_voice(text=text, instruct=instruct),
                        engine,
                        label=resume_key,
                        verbose=verbose,
                        reject_dir=get_reject_dir(workdir, "audition"),
                        metadata={
                            "phase": "audition",
                            "character": character,
                            "emotion": emotion,
                            "text": text,
                            "instruct": instruct,
                        },
                    )
                else:
                    audio, sr = engine.design_voice(text=text, instruct=instruct)
                sf.write(str(wav_path), audio, sr)
                from .audio import wav_sha256

                character, _, emotion = resume_key.partition("/")
                resume.update(
                    resume_key,
                    task_hash,
                    character=character,
                    emotion=emotion,
                    prompt=instruct,
                    audition_line=text,
                    seed=int(getattr(config, "seed", 0) or 0),
                    wav_sha256=wav_sha256(wav_path),
                )
                resume.save()
                generated_count += 1
            except Exception as e:
                print(f"failed to generate {resume_key}: {e}")
                raise

    resume.save()
    if generated_count == 0 and skipped_count > 0:
        print(f"audition: all {skipped_count} samples up to date.")
    else:
        print(f"audition: {generated_count} generated, {skipped_count} skipped")


def _resolve_emotion(instruction: str) -> str:
    """map a segment instruction to the closest known emotion key."""
    if not instruction:
        return "neutral"
    key = instruction.strip().lower()
    if key in VOICE_EMOTIONS:
        return key
    # fuzzy: check if any emotion key is a substring or vice versa
    for emotion in VOICE_EMOTIONS:
        if emotion in key or key in emotion:
            return emotion
    return "neutral"


def _format_segments_for_log(segments: List[ScriptSegment]) -> str:
    """format segments for logging."""
    lines = []
    for i, seg in enumerate(segments):
        lines.append(f"[{i}] {seg.speaker}: {seg.text}")
        if seg.instruction:
            lines.append(f"     instruction: {seg.instruction}")
    return "\n".join(lines)


def process_script_chunk_with_validation(
    text_chunk: str,
    characters_list: List[Character],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    verbose: bool = False,
) -> List[ScriptSegment]:
    """convert text chunk to script segments with iterative validation/fixing."""
    from .utils import log

    segments = process_script_chunk(
        text_chunk,
        characters_list,
        api_base,
        api_key,
        model,
        thinking_budget,
    )

    log(
        "VALIDATION_START",
        f"validating {len(segments)} segments",
        {
            "source_text": text_chunk,
            "segments": _format_segments_for_log(segments),
        },
    )

    total = VALIDATION_MAX_RETRIES + 1
    for attempt in range(1, total):
        result = validate_chunk(text_chunk, segments)
        if not result.missing and not result.hallucinated:
            log("VALIDATION_OK", f"passed on attempt {attempt}/{total}")
            if attempt > 1:
                print(f"    revise: attempt {attempt}/{total}: passed")
            return segments

        detail = format_validation_failure(result, segments, text_chunk)
        log(
            "VALIDATION_FAILED",
            f"attempt {attempt}/{total}",
            {
                "missing_count": str(len(result.missing)),
                "hallucinated_count": str(len(result.hallucinated)),
                "details": detail,
                "current_segments": _format_segments_for_log(segments),
            },
        )

        print(
            f"    revise: attempt {attempt}/{total}: "
            f"{len(result.missing)} missing, "
            f"{len(result.hallucinated)} hallucinated; fixing..."
        )
        for line in detail.split("\n"):
            print(f"      {line}")

        if result.hallucinated:
            _remove_hallucinations(segments, result.hallucinated)
            continue

        if result.missing:
            _fill_missing_fragments(
                segments,
                result.missing,
                text_chunk,
                characters_list,
                api_base,
                api_key,
                model,
                verbose,
                thinking_budget,
            )

    # final validation
    result = validate_chunk(text_chunk, segments)
    if result.missing or result.hallucinated:
        detail = format_validation_failure(result, segments, text_chunk)
        log(
            "VALIDATION_FINAL_FAILURE",
            f"failed after {total} attempts",
            {
                "missing_count": str(len(result.missing)),
                "hallucinated_count": str(len(result.hallucinated)),
                "details": detail,
                "final_segments": _format_segments_for_log(segments),
                "source_text": text_chunk,
            },
        )
        print(
            f"    revise: attempt {total}/{total}: "
            f"{len(result.missing)} missing, "
            f"{len(result.hallucinated)} hallucinated; giving up"
        )
        for line in detail.split("\n"):
            print(f"      {line}")
        raise ValidationError(
            f"validation failed after {VALIDATION_MAX_RETRIES} iterative fix attempts"
        )

    print(f"    revise: attempt {total}/{total}: passed")
    return segments


def run_script_generation(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    verbose: bool = False,
    force: bool = False,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    revise: bool = False,
) -> bool:
    """generate dramatized scripts for chapters incrementally.

    if revise=True, each chunk is reviewed against source text and
    retried with feedback on validation failures.
    """
    cast = load_cast(workdir)
    # Cast hash for dependency tracking
    # Only name and aliases affect the script generation prompt
    cast_hash = compute_hash(
        [
            {
                "name": c.name,
                "aliases": c.aliases,
            }
            for c in cast
        ]
    )

    resume = ResumeManager.for_command(workdir, "script", force=force)
    script_dir = get_command_dir(workdir, "script")
    extract_dir = get_command_dir(workdir, "extract")

    if not cast:
        if (get_command_dir(workdir, "cast") / CAST_FILE).exists():
            msg = "cast file found but contains no characters."
        else:
            msg = "no cast found. run 'cast' command first."
        print(msg)
        raise RuntimeError(msg)

    # collect chapters to process
    txt_files = sorted(extract_dir.glob(f"*{TXT_EXT}"))

    # Pre-scan to see what's done
    completed_count = 0
    to_process = []

    for txt_path in txt_files:
        try:
            chapter_num = int(txt_path.stem.split("_")[0])
        except ValueError:
            continue
        if chapters and chapter_num not in chapters:
            continue

        text = txt_path.read_text(encoding="utf-8")
        # Input hash depends on text and cast
        input_hash = compute_hash({"text": text, "cast_hash": cast_hash})

        script_path = script_dir / (txt_path.stem + SCRIPT_EXT)

        if (
            not force
            and script_path.exists()
            and resume.is_fresh(str(chapter_num), input_hash)
        ):
            completed_count += 1
        else:
            to_process.append((chapter_num, txt_path, script_path, text, input_hash))

    if not to_process:
        print(f"script: all {completed_count + len(to_process)} chapters up to date.")
        return True

    print(
        f"script: {len(to_process)} chapters to process, {completed_count} already complete"
    )

    total_segments = 0
    chapters_processed = 0

    for i, (chapter_num, txt_path, script_path, text, input_hash) in enumerate(
        to_process
    ):
        chunks = split_text_smart(text)
        total_chunks = len(chunks)

        # Load partial progress from state
        current_segments = []
        completed_chunks = 0
        partial = resume.get_partial(str(chapter_num))
        if (
            not force
            and partial
            and partial.get("hash") == input_hash
            and script_path.exists()
        ):
            completed_chunks = partial.get("chunks_done", 0)
            current_segments = load_script(script_path)

        if completed_chunks > 0:
            status = f"resuming at chunk {completed_chunks + 1}"
        else:
            status = "starting"

        print(
            f"  [{i + 1}/{len(to_process)}] {txt_path.name}: {status} ({total_chunks} chunks)"
        )

        for j in tqdm(
            range(completed_chunks, total_chunks),
            desc=f"    chapter {chapter_num}",
            unit="chunk",
            initial=completed_chunks,
            total=total_chunks,
        ):
            chunk_text_str = chunks[j]
            try:
                if revise:
                    chunk_segments = process_script_chunk_with_validation(
                        chunk_text_str,
                        cast,
                        api_base,
                        api_key,
                        model or DEFAULT_LLM_MODEL,
                        thinking_budget,
                        verbose=verbose,
                    )
                else:
                    chunk_segments = process_script_chunk(
                        chunk_text_str,
                        cast,
                        api_base,
                        api_key,
                        model or DEFAULT_LLM_MODEL,
                        thinking_budget,
                    )
                if verbose:
                    speakers = set(s.speaker for s in chunk_segments)
                    tqdm.write(
                        f"      chunk {j + 1}: generated {len(chunk_segments)} segments. "
                        f"Speakers: {', '.join(sorted(speakers))}"
                    )
                current_segments.extend(chunk_segments)
                save_script(script_path, current_segments)

                # Save intermediate progress to state
                resume.set_partial(
                    str(chapter_num),
                    {
                        "hash": input_hash,
                        "chunks_done": j + 1,
                    },
                )
                resume.save()

            except Exception as e:
                print(f"\n    chunk {j + 1} FAILED: {type(e).__name__}: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()
                raise

        # Mark as done (this also clears partial state)
        resume.update(str(chapter_num), input_hash)
        resume.save()

        total_segments += len(current_segments)
        chapters_processed += 1
        print(f"    -> {len(current_segments)} segments")

    print(f"done: {chapters_processed} chapters, {total_segments} total segments")

    return True


def run_performance(
    workdir: Path,
    chapters: list[int] | None = None,
    config: Any = None,
    pooled: bool = False,
    verbose: bool = False,
    force: bool = False,
    retake: bool = False,
    only_hashes: set[str] | None = None,
) -> None:
    """synthesize audio from scripts with segment-level resume."""

    cast = load_cast(workdir)
    if not cast:
        if (get_command_dir(workdir, "cast") / CAST_FILE).exists():
            msg = "cast file found but contains no characters."
        else:
            msg = "no cast found. run 'cast' command first."
        print(msg)
        raise RuntimeError(msg)

    # build cast map including aliases
    cast_map = {}
    for c in cast:
        cast_map[c.name] = c
        if c.aliases:
            for alias in c.aliases:
                cast_map[alias] = c

    voices_dir = get_command_dir(workdir, "audition")
    script_dir = get_command_dir(workdir, "script")
    perform_dir = get_command_dir(workdir, "perform")

    if not any(voices_dir.glob(f"*{WAV_EXT}")):
        msg = "no voices found. run 'audition' command first."
        print(msg)
        raise RuntimeError(msg)

    if config is None:
        from .tts import TTSConfig

        config = TTSConfig(model_name=BASE_MODEL)
    engine = create_tts_engine(config)

    # Collect all pending chapters first
    metadata = load_metadata(workdir)
    pending = [
        (s, t)
        for _, s, t in list_chapters(
            metadata,
            script_dir,
            perform_dir,
            chapters_filter=chapters,
            source_ext=SCRIPT_EXT,
            target_ext=WAV_EXT,
        )
    ]
    if not pending:
        msg = "perform: no scripts found."
        print(msg)
        raise RuntimeError(msg)

    # resume manager for assembly
    resume = ResumeManager.for_command(workdir, "perform", force=force)

    # always use pooled strategy for best performance/caching
    _perform_pooled(
        engine,
        pending,
        voices_dir,
        cast_map,
        resume=resume,
        force=force,
        verbose=verbose,
        retake=retake,
        only_hashes=only_hashes,
    )


def _perform_pooled(
    engine: Any,
    pending: list[tuple[Path, Path]],
    voices_dir: Path,
    cast_map: dict[str, Character],
    resume: ResumeManager | None = None,
    force: bool = False,
    verbose: bool = False,
    retake: bool = False,
    only_hashes: set[str] | None = None,
) -> None:
    """synthesize chapters using unified pooled batching and segment caching."""
    # Pre-calculate character hashes for stable identification
    char_hashes = {
        name: compute_hash(
            {
                "name": char.name,
                "description": char.description,
                "audition_line": char.audition_line,
            }
        )
        for name, char in cast_map.items()
    }

    introduce_dir = get_command_dir(voices_dir.parent, "introduce")

    # load audition state so we can tie each perform chunk to the exact ref
    # wav bytes that produced it: any regeneration of the audition (new seed,
    # swapped file) changes this sha and invalidates cached perform segments.
    audition_resume = ResumeManager.for_command(voices_dir.parent, "audition")
    audition_shas: dict[tuple[str, str], str] = {}
    for key, entry in audition_resume.state.items():
        if isinstance(entry, dict) and "wav_sha256" in entry:
            char_key, _, emo_key = key.partition("/")
            if char_key and emo_key:
                audition_shas[(char_key, emo_key)] = str(entry["wav_sha256"])

    chapter_data = []
    segments_dir = get_segments_dir(pending[0][1].parent)

    for txt_path, wav_path in pending:
        segments = load_script(txt_path)
        if not segments:
            continue

        chapter_tasks = []
        for script_idx, segment in enumerate(segments):
            # skip retained segments (section markers, chapter numbers, etc.)
            if segment.speaker in RETAINED_SPEAKERS:
                continue

            char_opt = cast_map.get(segment.speaker) or cast_map.get("Narrator")
            char_name = char_opt.name if char_opt else ""
            char_hash = char_hashes.get(char_name, "")

            # select emotion-variant audition clip
            emotion = _resolve_emotion(segment.instruction)
            emotion_file = (
                f"{char_opt.name}{EMOTION_SEP}{emotion}{WAV_EXT}" if char_opt else None
            )
            ref_audio_path = voices_dir / emotion_file if emotion_file else None
            # fall back to introduce base if emotion variant missing
            used_introduce_base = False
            if ref_audio_path and not ref_audio_path.exists() and char_opt:
                ref_audio_path = introduce_dir / f"{char_opt.name}{WAV_EXT}"
                used_introduce_base = True

            # ref_text must match what's spoken in the reference clip
            if used_introduce_base:
                ref_text = char_opt.audition_line if char_opt else None
            else:
                _, ref_text_default = VOICE_EMOTIONS.get(emotion, ("", ""))
                ref_text = ref_text_default or (
                    char_opt.audition_line if char_opt else None
                )

            ref_wav_sha = audition_shas.get((char_name, emotion), "")
            if not ref_wav_sha and ref_audio_path and ref_audio_path.exists():
                from .audio import wav_sha256

                ref_wav_sha = wav_sha256(ref_audio_path)

            seg_data = {
                "text": segment.text,
                "speaker": segment.speaker,
                "emotion": emotion,
                "char_hash": char_hash,
                "ref_wav_sha": ref_wav_sha,
            }

            text_chunks = (
                [
                    c
                    for c in chunk_text(segment.text, engine.config.chunk_size)
                    if c.strip()
                ]
                if len(segment.text) > engine.config.chunk_size
                else [segment.text]
            )

            for i, chunk in enumerate(text_chunks):
                chunk_hash = (
                    compute_hash({**seg_data, "chunk_idx": i, "chunk_text": chunk})
                    if len(text_chunks) > 1
                    else compute_hash(seg_data)
                )

                chapter_tasks.append(
                    AudioTask(
                        text=chunk,
                        segment_hash=chunk_hash,
                        segments_dir=segments_dir,
                        voice_ref_audio=ref_audio_path,
                        voice_ref_text=ref_text,
                        metadata={
                            "script_idx": script_idx,
                            "chunk_idx": i,
                            "script_path": str(txt_path),
                            "speaker": segment.speaker,
                            "emotion": emotion,
                        },
                    )
                )
        chapter_data.append((wav_path, chapter_tasks))

    process_audio_pipeline(
        engine,
        chapter_data,
        resume=resume,
        desc="performing segments",
        force=force,
        verbose=verbose,
        retake=retake,
        only_hashes=only_hashes,
    )


# CLI Command Wrappers


def cmd_cast(args):
    chapters = get_chapters(args)

    run_cast_generation(
        Path(args.workdir),
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
        verbose=args.verbose,
        force=args.force,
        thinking_budget=args.thinking_budget,
    )


def cmd_audition(args):
    from .utils import get_design_config

    config = get_design_config(args)
    run_auditions(
        Path(args.workdir),
        verbose=args.verbose,
        force=args.force,
        audition_line=getattr(args, "audition_line", None),
        config=config,
        callback=getattr(args, "callback", False),
    )


def cmd_script(args):
    from .utils import Logger

    chapters = get_chapters(args)
    workdir = Path(args.workdir)
    Logger.init(workdir)

    run_script_generation(
        workdir,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
        verbose=args.verbose,
        force=args.force,
        thinking_budget=args.thinking_budget,
        revise=getattr(args, "revise", False),
    )


def cmd_perform(args):
    from .utils import get_clone_config

    chapters = get_chapters(args)
    config = get_clone_config(args)

    run_performance(
        Path(args.workdir),
        chapters,
        config,
        args.pooled,
        verbose=args.verbose,
        force=args.force,
        retake=getattr(args, "retake", False),
    )


def _normalize_text(text: str) -> str:
    """normalize text for comparison by collapsing whitespace."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_boundary_quotes(text: str) -> str:
    """strip quotes and whitespace from text boundaries for comparison."""
    return text.strip(' \t\n"\'""')


def _tokenize_with_positions(text: str) -> List[tuple[str, int, int]]:
    """tokenize text into (word, start, end) tuples, ignoring punctuation."""
    tokens = []
    # match alphanumeric words only, treating hyphens/apostrophes as separators
    # this handles cases like "near-religious" vs "near religious" or "don't" vs "dont"
    for m in re.finditer(r"\w+", text):
        word = m.group().lower()
        if word:
            tokens.append((word, m.start(), m.end()))
    return tokens


def _find_text_in_source(
    needle: str, haystack: str, start_pos: int = 0
) -> tuple[int, int] | None:
    """find needle in haystack using token alignment.

    returns (start, end) positions in the original haystack, or None if not found.
    """
    needle_tokens = _tokenize_with_positions(needle)
    if not needle_tokens:
        return None
    needle_words = [t[0] for t in needle_tokens]

    # search a window of haystack starting from start_pos
    haystack_chunk = haystack[start_pos:]
    haystack_tokens = _tokenize_with_positions(haystack_chunk)
    if not haystack_tokens:
        return None
    haystack_words = [t[0] for t in haystack_tokens]

    matcher = difflib.SequenceMatcher(
        None, needle_words, haystack_words, autojunk=False
    )
    # find the best match for the needle words in the haystack
    match = matcher.find_longest_match(0, len(needle_words), 0, len(haystack_words))

    # we want a match that includes at least 70% of the needle tokens
    if match.size >= len(needle_words) * 0.7:
        start_char = haystack_tokens[match.b][1] + start_pos
        end_char = haystack_tokens[match.b + match.size - 1][2] + start_pos
        return (start_char, end_char)

    return None


@dataclass
class ValidationResult:
    """result of script validation for a chapter."""

    missing: list[
        tuple[str, int, int, str]
    ]  # (text, insertion_index, split_offset, full_line)
    hallucinated: list[int]  # indices of segments not found in source


def validate_chunk(source_text: str, segments: List[ScriptSegment]) -> ValidationResult:
    """validate that segments match source text for a single chunk.

    returns ValidationResult with missing text fragments and hallucinated segment indices.
    """
    return _validate_segments(source_text, segments)


def _validate_segments(
    source_text: str, segments: List[ScriptSegment]
) -> ValidationResult:
    """core validation logic shared by validate_chunk and validate_script."""
    if not segments:
        return ValidationResult(
            missing=[("no segments provided", 0, 0, "")], hallucinated=[]
        )

    source_tokens = _tokenize_with_positions(source_text)
    source_words = [t[0] for t in source_tokens]

    script_words = []
    script_token_info = []  # (seg_idx, start, end)
    segment_stats = {}  # seg_idx -> {'total': 0, 'matched': 0}

    for i, seg in enumerate(segments):
        seg_tokens = _tokenize_with_positions(seg.text)
        segment_stats[i] = {"total": len(seg_tokens), "matched": 0}
        for t in seg_tokens:
            script_words.append(t[0])
            script_token_info.append((i, t[1], t[2]))

    matcher = difflib.SequenceMatcher(None, source_words, script_words, autojunk=False)
    opcodes = matcher.get_opcodes()

    missing_ranges = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for j in range(j1, j2):
                seg_idx = script_token_info[j][0]
                segment_stats[seg_idx]["matched"] += 1
        elif tag in ("delete", "replace"):
            if i1 < i2:
                start_char = source_tokens[i1][1]
                end_char = source_tokens[i2 - 1][2]

                if j1 < len(script_words):
                    ins_idx, split_offset, _ = script_token_info[j1]
                else:
                    ins_idx = len(segments)
                    split_offset = 0

                missing_ranges.append((start_char, end_char, ins_idx, split_offset))

    # merge contiguous missing ranges
    missing_fragments = []
    if missing_ranges:
        missing_ranges.sort()
        merged = [missing_ranges[0]]

        for current_start, current_end, current_ins, current_offset in missing_ranges[
            1:
        ]:
            last_start, last_end, last_ins, last_offset = merged[-1]
            gap_text = source_text[last_end:current_start]

            if (
                current_ins == last_ins
                and current_offset == last_offset
                and not re.search(r"\w", gap_text)
            ):
                merged[-1] = (last_start, current_end, last_ins, last_offset)
            else:
                merged.append((current_start, current_end, current_ins, current_offset))

        for start, end, ins_idx, split_offset in merged:
            while start > 0 and source_text[start - 1] in ".,;:?!\"'()[]-":
                start -= 1
            while end < len(source_text) and source_text[end] in ".,;:?!\"'()[]-":
                end += 1

            text = source_text[start:end].strip()

            line_start = source_text.rfind("\n", 0, start) + 1
            line_end = source_text.find("\n", end)
            if line_end == -1:
                line_end = len(source_text)
            full_line = source_text[line_start:line_end].strip()

            if len(text) > 1 or (len(text) == 1 and text.isalnum()):
                missing_fragments.append((text, ins_idx, split_offset, full_line))

    # identify hallucinated segments
    hallucinated_indices = []
    for i in range(len(segments)):
        stats = segment_stats[i]
        if stats["total"] == 0:
            continue
        ratio = stats["matched"] / stats["total"]
        if ratio < 0.5:
            hallucinated_indices.append(i)

    return ValidationResult(
        missing=missing_fragments, hallucinated=hallucinated_indices
    )


def validate_script(txt_path: Path, script_path: Path) -> ValidationResult:
    """validate that script segments match the source text using fuzzy diffing."""
    segments = load_script(script_path)
    if not segments:
        return ValidationResult(
            missing=[(f"no script found for {txt_path.name}", 0, 0, "")],
            hallucinated=[],
        )

    original_text = txt_path.read_text(encoding="utf-8")
    return _validate_segments(original_text, segments)


def _truncate(text: str, max_len: int = 80) -> str:
    """truncate text for display, adding ellipsis if needed."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_validation_failure(
    result: ValidationResult,
    segments: List[ScriptSegment],
    source_text: str,
) -> str:
    """format validation failures with detailed troubleshooting info."""
    if not result.missing and not result.hallucinated:
        return ""

    lines = []

    for i, (fragment, idx, offset, full_line) in enumerate(result.missing, 1):
        lines.append(f'[missing #{i}] "{_truncate(fragment, 60)}"')
        if full_line and full_line != fragment:
            lines.append(f'  full line: "{_truncate(full_line, 70)}"')

        # show insertion context
        if idx == 0:
            lines.append("  insert at: beginning of script")
        elif idx >= len(segments):
            if segments:
                prev_seg = segments[-1]
                lines.append(
                    f"  insert after segment {len(segments) - 1}: "
                    f'{prev_seg.speaker}: "{_truncate(prev_seg.text, 50)}"'
                )
            else:
                lines.append("  insert at: end of empty script")
        else:
            prev_seg = segments[idx - 1]
            next_seg = segments[idx]
            lines.append(
                f"  insert after segment {idx - 1}: "
                f'{prev_seg.speaker}: "{_truncate(prev_seg.text, 50)}"'
            )
            lines.append(
                f"  insert before segment {idx}: "
                f'{next_seg.speaker}: "{_truncate(next_seg.text, 50)}"'
            )

        if offset > 0:
            lines.append(f"  split offset: {offset} chars into segment {idx}")

    for idx in result.hallucinated:
        if idx < len(segments):
            seg = segments[idx]
            lines.append(
                f'[hallucinated #{idx}] {seg.speaker}: "{_truncate(seg.text, 60)}"'
            )
            lines.append("  no matching text found in source")

    return "\n".join(lines)


def run_validation(
    workdir: Path,
    chapters: list[int] | None = None,
    check_missing: bool = True,
    check_hallucinated: bool = True,
) -> dict[str, ValidationResult]:
    """validate scripts against source text for all chapters.

    returns a dict mapping chapter names to ValidationResult.
    """

    extract_dir = get_command_dir(workdir, "extract")
    script_dir = get_command_dir(workdir, "script")

    txt_files = sorted(extract_dir.glob(f"*{TXT_EXT}"))
    if not txt_files:
        print("no text files found in extract/!")
        return {}

    # filter to relevant chapters
    chapters_to_check = []
    for txt_path in txt_files:
        try:
            chapter_num = int(txt_path.stem.split("_")[0])
        except ValueError:
            continue
        if chapters and chapter_num not in chapters:
            continue
        script_path = script_dir / (txt_path.stem + SCRIPT_EXT)
        if not script_path.exists():
            continue
        chapters_to_check.append((txt_path, script_path))

    if not chapters_to_check:
        print("no chapters with scripts to validate")
        return {}

    results = {}
    total_missing = 0
    total_hallucinated = 0

    for txt_path, script_path in tqdm(
        chapters_to_check, desc="validating", unit="chapter"
    ):
        result = validate_script(txt_path, script_path)
        results[txt_path.name] = result

        if check_missing:
            total_missing += len(result.missing)
        if check_hallucinated:
            total_hallucinated += len(result.hallucinated)

    # print results
    print()
    for txt_path, script_path in chapters_to_check:
        result = results[txt_path.name]
        issues = []

        if check_missing and result.missing:
            issues.append(f"{len(result.missing)} missing")
        if check_hallucinated and result.hallucinated:
            issues.append(f"{len(result.hallucinated)} hallucinated")

        if issues:
            print(f"\n{txt_path.name}: {', '.join(issues)}")

            # filter result based on what we're checking
            filtered_result = ValidationResult(
                missing=result.missing if check_missing else [],
                hallucinated=result.hallucinated if check_hallucinated else [],
            )
            segments = load_script(script_path)
            source_text = txt_path.read_text(encoding="utf-8")
            detail = format_validation_failure(filtered_result, segments, source_text)
            if detail:
                for line in detail.split("\n"):
                    print(f"  {line}")
        else:
            print(f"{txt_path.name}: OK")

    # summary
    summary_parts = []
    if check_missing:
        summary_parts.append(f"{total_missing} missing")
    if check_hallucinated:
        summary_parts.append(f"{total_hallucinated} hallucinated")

    if total_missing == 0 and total_hallucinated == 0:
        print(f"\nvalidate: all {len(results)} chapters OK")
    else:
        msg = (
            f"validate: found {', '.join(summary_parts)} across {len(results)} chapters"
        )
        print(f"\n{msg}")
        raise ValidationError(msg)

    return results


def cmd_revise(args):
    """review and repair scripts.

    --dry-run: only report missing/hallucinated segments (validate).
    otherwise: fix them via LLM and hallucination removal."""
    from .utils import Logger

    chapters = get_chapters(args)
    workdir = Path(args.workdir)
    Logger.init(workdir)

    if args.dry_run:
        run_validation(workdir, chapters, True, True)
        return

    # --prune: do only the local destructive step (strip hallucinations),
    # skip the expensive LLM fix-missing pass.
    fix_missing = not getattr(args, "prune", False)

    run_revise(
        workdir,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
        fix_missing=fix_missing,
        fix_hallucinated=True,
        verbose=args.verbose,
        thinking_budget=args.thinking_budget,
    )


def _attempt_merge(segments: List[ScriptSegment], index: int) -> bool:
    """merge segment at index with next segment if speakers match.

    returns True if merged (and list is shortened).
    """
    if index < 0 or index >= len(segments) - 1:
        return False

    s1 = segments[index]
    s2 = segments[index + 1]

    if s1.speaker == s2.speaker:
        # merge s2 into s1
        s1.text = s1.text.rstrip() + " " + s2.text.lstrip()
        # keep s1's instruction as the primary one
        segments.pop(index + 1)
        return True
    return False


def _remove_hallucinations(
    segments: List[ScriptSegment], hallucinated_indices: List[int]
) -> int:
    """remove segments identified as hallucinations."""
    removed = 0
    for idx in sorted(hallucinated_indices, reverse=True):
        seg = segments[idx]
        print(f"  removing [{idx}] {seg.speaker}: {seg.text}")
        del segments[idx]
        removed += 1
    return removed


def _segments_to_context(segments: List[ScriptSegment], start: int, end: int) -> str:
    """format a slice of segments as JSON for LLM context."""
    start = max(0, start)
    end = min(len(segments), end)
    context = [
        {"speaker": s.speaker, "text": s.text, "instruction": s.instruction}
        for s in segments[start:end]
    ]
    return json.dumps(context, ensure_ascii=False, indent=2) if context else "[]"


def _fill_missing_fragments(
    segments: List[ScriptSegment],
    missing: list[tuple[str, int, int, str]],
    original_text: str,
    cast: List[Character],
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    verbose: bool,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> int:
    """fill missing text fragments using LLM with surrounding script as context."""
    added = 0
    context_segs = 3  # number of surrounding segments to include as context

    # sort descending so insertions don't invalidate subsequent indices
    for fragment, insertion_idx, split_offset, full_line in sorted(
        missing, key=lambda x: (x[1], x[2]), reverse=True
    ):
        if verbose:
            print(
                f"\n    missing fragment (@ {insertion_idx}+{split_offset}): {full_line}"
            )

        target_idx = insertion_idx
        if split_offset > 0 and insertion_idx < len(segments):
            seg = segments[insertion_idx]
            if split_offset < len(seg.text):
                from copy import deepcopy

                left_seg, right_seg = deepcopy(seg), deepcopy(seg)
                left_seg.text = seg.text[:split_offset].rstrip()
                right_seg.text = seg.text[split_offset:].lstrip()
                segments[insertion_idx] = left_seg
                segments.insert(insertion_idx + 1, right_seg)
                target_idx = insertion_idx + 1

        # use surrounding script JSON as context
        context_before = _segments_to_context(
            segments, target_idx - context_segs, target_idx
        )
        context_after = _segments_to_context(
            segments, target_idx, target_idx + context_segs
        )

        try:
            new_segs = fix_missing_segment(
                fragment,
                context_before,
                context_after,
                cast,
                api_base,
                api_key,
                model or DEFAULT_LLM_MODEL,
                thinking_budget,
            )
            if new_segs:
                for j, s in enumerate(new_segs):
                    segments.insert(target_idx + j, s)
                # merge neighbors
                _attempt_merge(segments, target_idx + len(new_segs) - 1)
                for j in range(len(new_segs) - 2, -1, -1):
                    _attempt_merge(segments, target_idx + j)
                if target_idx > 0:
                    _attempt_merge(segments, target_idx - 1)
                added += len(new_segs)
        except Exception as e:
            print(f"    failed: {e}")
            raise
    return added


def run_revise(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    fix_missing: bool = True,
    fix_hallucinated: bool = True,
    verbose: bool = False,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> None:
    """fix script issues by filling missing segments and removing hallucinated ones."""
    cast = load_cast(workdir)
    extract_dir, script_dir = (
        get_command_dir(workdir, "extract"),
        get_command_dir(workdir, "script"),
    )
    txt_files = sorted(extract_dir.glob(f"*{TXT_EXT}"))
    if not txt_files:
        msg = "revise: no text files found in extract/!"
        print(msg)
        raise RuntimeError(msg)

    total_added, total_removed = 0, 0
    for txt_path in txt_files:
        try:
            num = int(txt_path.stem.split("_")[0])
        except ValueError:
            continue
        if chapters and num not in chapters:
            continue
        script_path = script_dir / (txt_path.stem + SCRIPT_EXT)
        if not script_path.exists():
            continue

        result = validate_script(txt_path, script_path)
        if not result.missing and not result.hallucinated:
            continue

        segments = load_script(script_path)
        if fix_hallucinated and result.hallucinated:
            print(
                f"\n{txt_path.name}: removing {len(result.hallucinated)} hallucination(s)..."
            )
            total_removed += _remove_hallucinations(segments, result.hallucinated)
            save_script(script_path, segments)

        if fix_missing:
            result = validate_script(txt_path, script_path)  # re-validate
            if result.missing:
                print(
                    f"\n{txt_path.name}: filling {len(result.missing)} missing fragment(s)..."
                )
                total_added += _fill_missing_fragments(
                    segments,
                    result.missing,
                    txt_path.read_text(encoding="utf-8"),
                    cast,
                    api_base,
                    api_key,
                    model,
                    verbose,
                    thinking_budget,
                )
                save_script(script_path, segments)
                # strict: after the llm fix pass, any remaining missing
                # fragments are unrecoverable — raise rather than ship a gap.
                final = validate_script(txt_path, script_path)
                if final.missing:
                    raise RuntimeError(
                        f"revise: {txt_path.name} still has "
                        f"{len(final.missing)} missing fragment(s) after "
                        f"llm fix attempt"
                    )

    if total_added > 0 or total_removed > 0:
        print(f"\nrevise: added {total_added}, removed {total_removed} segment(s)")
    else:
        print("revise: no issues found.")

    # summary
    summary_parts = []
    if fix_missing and total_added > 0:
        summary_parts.append(f"added {total_added} segment(s)")
    if fix_hallucinated and total_removed > 0:
        summary_parts.append(f"removed {total_removed} segment(s)")

    if summary_parts:
        print(f"\nrevise: {', '.join(summary_parts)}")
    else:
        print("revise: no issues found.")


def _step_if_changed(step: bool, phase: str, path: Path, before: float) -> None:
    """raise StepComplete if step mode is active and files changed."""
    from .utils import dir_mtime

    if step and dir_mtime(path) > before:
        from .main import StepComplete

        raise StepComplete(phase)


def dramatize_book(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    design_config: Any = None,
    clone_config: Any = None,
    pooled: bool = False,
    verbose: bool = False,
    force: bool = False,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    revise: bool = False,
    step: bool = False,
    redo_phase: str | None = None,
    retake: bool = False,
    callback: bool = False,
) -> None:
    """run full dramatization pipeline."""
    before = dir_mtime(get_command_dir(workdir, "cast"))
    run_cast_generation(
        workdir,
        api_base,
        api_key,
        model,
        chapters,
        verbose=verbose,
        force=force or redo_phase == "cast",
        thinking_budget=thinking_budget,
    )
    _step_if_changed(step, "cast", get_command_dir(workdir, "cast"), before)

    from .introduce import run_introduce

    before = dir_mtime(get_command_dir(workdir, "introduce"))
    run_introduce(
        workdir,
        verbose=verbose,
        force=force or redo_phase == "introduce",
        config=design_config,
        callback=callback,
    )
    _step_if_changed(step, "introduce", get_command_dir(workdir, "introduce"), before)

    before = dir_mtime(get_command_dir(workdir, "audition"))
    run_auditions(
        workdir,
        verbose=verbose,
        force=force or redo_phase == "audition",
        config=design_config,
        callback=callback,
    )
    _step_if_changed(step, "audition", get_command_dir(workdir, "audition"), before)

    before = dir_mtime(get_command_dir(workdir, "script"))
    run_script_generation(
        workdir,
        api_base,
        api_key,
        model,
        chapters,
        verbose=verbose,
        force=force or redo_phase == "script",
        thinking_budget=thinking_budget,
        revise=revise,
    )
    _step_if_changed(step, "script", get_command_dir(workdir, "script"), before)

    before = dir_mtime(get_command_dir(workdir, "script"))
    run_revise(
        workdir,
        api_base,
        api_key,
        model,
        chapters,
        verbose=verbose,
        thinking_budget=thinking_budget,
    )
    _step_if_changed(step, "revise", get_command_dir(workdir, "script"), before)

    before = dir_mtime(get_command_dir(workdir, "perform"))
    run_performance(
        workdir,
        chapters,
        clone_config,
        pooled,
        verbose=verbose,
        force=force or redo_phase == "perform",
        retake=retake,
    )
    _step_if_changed(step, "perform", get_command_dir(workdir, "perform"), before)

    before = dir_mtime(get_command_dir(workdir, "perform") / "segments")
    from .retake import run_retake

    run_retake(
        workdir,
        command="perform",
        chapters=chapters,
        config=clone_config,
        verbose=verbose,
    )
    _step_if_changed(
        step, "retake", get_command_dir(workdir, "perform") / "segments", before
    )
