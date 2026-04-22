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
    CAST_CHUNK_OVERLAP_WORDS,
    CAST_CHUNK_WORDS,
    CAST_FILE,
    DEFAULT_CAST,
    DEFAULT_LLM_MODEL,
    DEFAULT_THINKING_BUDGET,
    EMOTION_SEP,
    RETAINED_SPEAKERS,
    REVIEW_BATCH_SIZE,
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
    review_script_batch,
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


_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’"}


def _strip_wrapping_quotes(text: str) -> str:
    """strip a single matched pair of boundary quotes, if present.

    LLM segments split narrator from dialogue, so a segment wrapped in
    matching quotes has redundant boundary marks — spoken as a pause or
    glottal click by TTS. preserves inner quotes and unmatched boundaries.
    """
    s = text.strip()
    if len(s) < 2:
        return text
    close = _QUOTE_PAIRS.get(s[0])
    if close and s.endswith(close):
        return s[1:-1].strip()
    return text


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
                "text": _strip_wrapping_quotes(s.text),
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

        # drop proposed aliases that conflict with other characters in the cast
        existing_low = existing.name.lower()
        rejected_aliases: list[str] = []
        filtered_new: set[str] = set()
        for a in new_aliases:
            a_low = a.lower()
            owner = cast_map.get(a_low)
            if owner is not None and owner is not existing:
                rejected_aliases.append(f"{a} (conflicts with '{owner.name}')")
                continue
            alias_owner = alias_map.get(a_low)
            if alias_owner is not None and alias_owner != existing_low:
                rejected_aliases.append(
                    f"{a} (already alias of '{cast_map[alias_owner].name}')"
                )
                continue
            filtered_new.add(a)
        new_aliases = filtered_new

        added_aliases = sorted(new_aliases - old_aliases)
        if new_aliases != set(existing.aliases or []):
            existing.aliases = sorted(new_aliases) if new_aliases else None
        if added_aliases:
            diff_parts.append("+aliases: " + ", ".join(repr(a) for a in added_aliases))
        if verbose and rejected_aliases:
            diff_parts.append("rejected aliases: " + ", ".join(rejected_aliases))
        # refresh alias_map for newly added aliases
        for a in added_aliases:
            alias_map[a.lower()] = existing_low

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
        print(f"  added new character: '{c.name}'")
        if clean_aliases:
            print(f"    aliases: {', '.join(repr(a) for a in clean_aliases)}")
        if c.description:
            print(f"    description: {c.description!r}")
        if c.audition_line:
            print(f"    audition_line: {c.audition_line!r}")
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


def _chunk_by_words(text: str, max_words: int, overlap_words: int) -> list[str]:
    """split text at paragraph boundaries with tail-overlap for coreference.

    each chunk (after the first) is prefixed with the last `overlap_words`
    from the prior chunk so a character introduced near a boundary remains
    attributable in the next chunk.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0

    def _flush() -> None:
        if current:
            chunks.append("\n\n".join(current))

    for p in paragraphs:
        wc = len(p.split())
        if count + wc > max_words and current:
            _flush()
            # build overlap from tail of just-flushed chunk
            tail_words: list[str] = []
            for prev in reversed(current):
                tail_words = prev.split() + tail_words
                if len(tail_words) >= overlap_words:
                    break
            overlap = " ".join(tail_words[-overlap_words:]) if overlap_words else ""
            current = [overlap] if overlap else []
            count = len(overlap.split())
        current.append(p)
        count += wc

    _flush()
    return chunks


def _save_cast_narrator_first(workdir: Path, cast_map: dict[str, Character]) -> None:
    """persist cast with 'narrator' pinned to the front if present."""
    final_cast = list(cast_map.values())
    narrator = next((c for c in final_cast if c.name.lower() == "narrator"), None)
    if narrator:
        final_cast.remove(narrator)
        final_cast.insert(0, narrator)
    save_cast(workdir, final_cast)


def _batch_chunks(batch_chapters: list[int], chapter_map: dict[int, Path]) -> list[str]:
    """build the sample text for a batch and split it into cast chunks."""
    full_sample = ""
    for num in batch_chapters:
        txt_path = chapter_map[num]
        full_sample += f"\n--- Chapter {txt_path.stem} ---\n"
        full_sample += txt_path.read_text(encoding="utf-8")
    return _chunk_by_words(full_sample, CAST_CHUNK_WORDS, CAST_CHUNK_OVERLAP_WORDS)


def _process_cast_batch(
    workdir: Path,
    batch_chapters: list[int],
    chapter_map: dict[int, Path],
    cast_map: dict[str, Character],
    alias_map: dict[str, str],
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    verbose: bool,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    progress: Any = None,
) -> int:
    """process a single batch of chapters, chunked for coreference context."""
    chunks = _batch_chunks(batch_chapters, chapter_map)
    total_found = 0
    for ci, chunk in enumerate(chunks):
        if progress is not None:
            progress.set_postfix_str(
                f"batch {batch_chapters} chunk {ci + 1}/{len(chunks)} "
                f"({len(cast_map)} known)"
            )
        elif len(chunks) > 1:
            print(
                f"    chunk {ci + 1}/{len(chunks)} "
                f"(~{len(chunk.split())} words, {len(cast_map)} known)"
            )
        current_cast = list(cast_map.values())
        summary = "\n".join(
            f"- {c.name}: {c.description}"
            + (f" (also known as: {', '.join(c.aliases)})" if c.aliases else "")
            for c in current_cast
        )
        chunk_cast = generate_cast(
            chunk,
            api_base,
            api_key,
            model or DEFAULT_LLM_MODEL,
            existing_cast_summary=summary,
            thinking_budget=thinking_budget,
        )
        for c in chunk_cast:
            _merge_character_into_cast(c, cast_map, alias_map, verbose=verbose)
        total_found += len(chunk_cast)
        _save_cast_narrator_first(workdir, cast_map)
        if progress is not None:
            progress.update(1)

    return total_found


def run_cast_generation(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    verbose: bool = False,
    force: bool = False,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    accept: bool = False,
) -> List[Character]:
    """analyze book and generate cast list.

    when `accept` is True, skip llm analysis and mark the existing cast as
    fresh under current chapter hashes. useful after hand-editing
    cast/characters.json to lock it in without re-analyzing.
    """
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

    if accept:
        for num, h in chapter_hashes.items():
            resume.update(str(num), h)
        resume.save()
        print(f"cast: accepted existing cast for {len(chapter_hashes)} chapter(s)")
        return existing_cast

    print(f"cast: analyzing {len(chapters_to_process)} chapters...")

    cast_map = {c.name.lower(): c for c in existing_cast}
    alias_map = {
        a.lower(): c.name.lower() for c in existing_cast if c.aliases for a in c.aliases
    }

    batch_size = CAST_BATCH_SIZE
    batches = [
        chapters_to_process[i : i + batch_size]
        for i in range(0, len(chapters_to_process), batch_size)
    ]
    # precompute total chunks across all batches so the progress bar is accurate
    total_chunks = sum(len(_batch_chunks(b, chapter_map)) for b in batches)
    progress = tqdm(total=total_chunks, desc="cast", unit="chunk")

    try:
        for batch_chapters in batches:
            _process_cast_batch(
                workdir,
                batch_chapters,
                chapter_map,
                cast_map,
                alias_map,
                api_base,
                api_key,
                model,
                verbose,
                thinking_budget,
                progress=progress,
            )

            for num in batch_chapters:
                resume.update(str(num), chapter_hashes[num])
            resume.save()
            _save_cast_narrator_first(workdir, cast_map)
    finally:
        progress.close()

    return list(cast_map.values())


def _emote_tasks(
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


def _accept_existing_emotes(
    workdir: Path,
    cast: List[Character],
    voices_dir: Path,
    resume: ResumeManager,
    audition_line: str | None,
    preset_map: dict[str, str],
    preset_voices: bool,
) -> None:
    """mark existing emote wavs as fresh under current hashes, without regen."""
    from .audio import wav_sha256
    from .audition import recorded_seed

    updated = 0
    missing: list[str] = []
    for char in cast:
        intro_seed = recorded_seed(workdir, char.name)
        voice_id = preset_map.get(char.name) if preset_voices else None
        if preset_voices and not voice_id:
            missing.append(f"{char.name} (no voice_id)")
            continue
        for filename, resume_key, text, instruct in _emote_tasks(char, audition_line):
            wav_path = voices_dir / f"{filename}{WAV_EXT}"
            if not wav_path.exists():
                missing.append(resume_key)
                continue
            emote_instruct = instruct
            if preset_voices:
                _, _, emotion_only = instruct.partition("; ")
                emote_instruct = emotion_only or instruct
            task_hash = compute_hash(
                {
                    "name": char.name,
                    "description": char.description,
                    "text": text,
                    "instruct": emote_instruct,
                    "audition_seed": intro_seed,
                    "voice_id": voice_id or "",
                }
            )
            character, _, emotion = resume_key.partition("/")
            prior = resume.state.get(resume_key, {})
            prior_seed = (
                int(prior.get("seed", 0) or 0) if isinstance(prior, dict) else 0
            )
            resume.update(
                resume_key,
                task_hash,
                character=character,
                emotion=emotion,
                prompt=emote_instruct,
                audition_line=text,
                seed=prior_seed,
                wav_sha256=wav_sha256(wav_path),
            )
            updated += 1
    resume.save()
    print(f"emote: accepted {updated} existing samples")
    if missing:
        print(
            f"emote: {len(missing)} missing — re-run without --accept "
            f"to fill in: {', '.join(missing)}"
        )


def run_emotes(
    workdir: Path,
    cast: List[Character] | None = None,
    verbose: bool = False,
    force: bool = False,
    audition_line: str | None = None,
    config: Any = None,
    callback: bool = False,
    preset_voices: bool = False,
    accept: bool = False,
) -> None:
    """generate voice samples for cast with emotion variants.

    when preset_voices=True, uses backend voice ids (from audition/voices.json)
    and the `instructions` feature to synthesize emotion variants for inspection
    only; perform does not read these wavs in preset mode.
    """

    if cast is None:
        cast = load_cast(workdir)

    voices_dir = get_command_dir(workdir, "emote")
    resume = ResumeManager.for_command(workdir, "emote", force=force)

    from .audition import recorded_seed

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

    preset_map: dict[str, str] = {}
    if preset_voices:
        from .casting import load_voices

        preset_map = load_voices(workdir)
        if not preset_map:
            print("emote: --preset-voices set but no audition/voices.json; skipping")
            return

    if accept:
        _accept_existing_emotes(
            workdir, cast, voices_dir, resume, audition_line, preset_map, preset_voices
        )
        return

    if config is None:
        from .tts import TTSConfig

        config = TTSConfig(model_name=VOICE_DESIGN_MODEL)
    engine = create_tts_engine(config)

    print(
        f"generating emotes for {len(cast)} characters "
        f"({len(VOICE_EMOTIONS)} emotions each)..."
    )

    generated_count = 0
    skipped_count = 0

    for char in tqdm(cast, desc="casting voices"):
        tasks = _emote_tasks(char, audition_line)

        voice_id = preset_map.get(char.name) if preset_voices else None
        if preset_voices and not voice_id:
            continue

        # reuse the per-character seed recorded by audition so all of a
        # character's ref clips (base + emotion variants) ride the same
        # trajectory. a changed audition seed forces re-emote via the hash.
        intro_seed = recorded_seed(workdir, char.name)
        if intro_seed > 0:
            engine.config.seed = intro_seed

        for filename, resume_key, text, instruct in tasks:
            wav_path = voices_dir / f"{filename}{WAV_EXT}"

            # in preset mode, drop the character description from instruct
            # (the voice_id supplies the character identity).
            emote_instruct = instruct
            if preset_voices:
                _, _, emotion_only = instruct.partition("; ")
                emote_instruct = emotion_only or instruct

            task_data = {
                "name": char.name,
                "description": char.description,
                "text": text,
                "instruct": emote_instruct,
                "audition_seed": intro_seed,
                "voice_id": voice_id or "",
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

            def _generate():
                if preset_voices:
                    return engine.synthesize(text, emote_instruct, speaker=voice_id)
                return engine.design_voice(text=text, instruct=emote_instruct)

            try:
                if callback:
                    from .callback import generate_with_callback
                    from .retake import get_reject_dir

                    character, _, emotion = resume_key.partition("/")
                    audio, sr = generate_with_callback(
                        _generate,
                        engine,
                        label=resume_key,
                        verbose=verbose,
                        reject_dir=get_reject_dir(workdir, "emote"),
                        metadata={
                            "phase": "emote",
                            "character": character,
                            "emotion": emotion,
                            "text": text,
                            "instruct": emote_instruct,
                        },
                    )
                else:
                    audio, sr = _generate()
                sf.write(str(wav_path), audio, sr)
                from .audio import wav_sha256

                character, _, emotion = resume_key.partition("/")
                resume.update(
                    resume_key,
                    task_hash,
                    character=character,
                    emotion=emotion,
                    prompt=emote_instruct,
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
        print(f"emote: all {skipped_count} samples up to date.")
    else:
        print(f"emote: {generated_count} generated, {skipped_count} skipped")


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
    accept: bool = False,
) -> bool:
    """generate dramatized scripts for chapters incrementally.

    if revise=True, each chunk is reviewed against source text and
    retried with feedback on validation failures.

    if accept=True, skip llm generation and mark existing script files as
    fresh under the current (extract + cast) input hash. useful after hand-
    editing a script to lock it in without re-generating.
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

    if accept:
        accepted, missing = 0, []
        for chapter_num, _txt_path, script_path, _text, input_hash in to_process:
            if not script_path.exists():
                missing.append(chapter_num)
                continue
            resume.clear_partial(str(chapter_num))
            resume.update(str(chapter_num), input_hash)
            accepted += 1
        resume.save()
        print(f"script: accepted {accepted} existing script(s)")
        if missing:
            print(
                f"script: {len(missing)} missing — re-run without --accept "
                f"to generate: {missing}"
            )
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
    accept: bool = False,
) -> None:
    """synthesize audio from scripts with segment-level resume.

    accept=True skips synthesis; existing chapter wavs are re-stamped as fresh
    under the current segment-content fingerprint.
    """
    if accept:
        print("perform: skipped (--accept)")
        return

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

    voices_dir = get_command_dir(workdir, "emote")
    script_dir = get_command_dir(workdir, "script")
    perform_dir = get_command_dir(workdir, "perform")

    # perform can use emote wavs, fall back to audition wavs, or run in preset
    # mode (voices.json assigns backend voice_ids so no ref wavs are needed).
    from .casting import load_voices

    has_emote = any(voices_dir.glob(f"*{WAV_EXT}"))
    audition_dir = get_command_dir(workdir, "audition")
    has_audition = any(audition_dir.glob(f"*{WAV_EXT}"))
    has_preset = bool(load_voices(workdir))
    if not (has_emote or has_audition or has_preset):
        msg = "no voices found. run 'audition' (and optionally 'emote') first."
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

    audition_dir = get_command_dir(voices_dir.parent, "audition")

    # preset-voice mode: audition/voices.json assigns each character a backend
    # voice_id. perform skips cloning and speaks via voice_id + instructions.
    from .casting import load_voices

    preset_voices = load_voices(voices_dir.parent)

    # load emote state so we can tie each perform chunk to the exact ref
    # wav bytes that produced it: any regeneration of the emote (new seed,
    # swapped file) changes this sha and invalidates cached perform segments.
    emote_resume = ResumeManager.for_command(voices_dir.parent, "emote")
    emote_shas: dict[tuple[str, str], str] = {}
    for key, entry in emote_resume.state.items():
        if isinstance(entry, dict) and "wav_sha256" in entry:
            char_key, _, emo_key = key.partition("/")
            if char_key and emo_key:
                emote_shas[(char_key, emo_key)] = str(entry["wav_sha256"])

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

            # select emotion variant
            emotion = _resolve_emotion(segment.instruction)
            preset_voice_id = preset_voices.get(char_name) if preset_voices else None
            emotion_instruct, _ = VOICE_EMOTIONS.get(emotion, ("", ""))

            if preset_voice_id:
                # preset mode: no ref audio, just voice_id + emotion instruction
                ref_audio_path = None
                ref_text = None
                ref_wav_sha = ""
                seg_data = {
                    "text": segment.text,
                    "speaker": segment.speaker,
                    "emotion": emotion,
                    "char_hash": char_hash,
                    "preset_voice": preset_voice_id,
                    "instruct": emotion_instruct,
                }
            else:
                emotion_file = (
                    f"{char_opt.name}{EMOTION_SEP}{emotion}{WAV_EXT}"
                    if char_opt
                    else None
                )
                ref_audio_path = voices_dir / emotion_file if emotion_file else None
                # fall back to audition base if emotion variant missing
                used_audition_base = False
                if ref_audio_path and not ref_audio_path.exists() and char_opt:
                    ref_audio_path = audition_dir / f"{char_opt.name}{WAV_EXT}"
                    used_audition_base = True

                # ref_text must match what's spoken in the reference clip
                if used_audition_base:
                    ref_text = char_opt.audition_line if char_opt else None
                else:
                    _, ref_text_default = VOICE_EMOTIONS.get(emotion, ("", ""))
                    ref_text = ref_text_default or (
                        char_opt.audition_line if char_opt else None
                    )

                ref_wav_sha = emote_shas.get((char_name, emotion), "")
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
                        instruct=(emotion_instruct if preset_voice_id else ""),
                        preset_voice=preset_voice_id,
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
        accept=getattr(args, "accept", False),
    )


def cmd_emote(args):
    from .utils import get_design_config

    config = get_design_config(args)
    run_emotes(
        Path(args.workdir),
        verbose=args.verbose,
        force=args.force,
        audition_line=getattr(args, "audition_line", None),
        config=config,
        callback=getattr(args, "callback", False),
        accept=getattr(args, "accept", False),
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
        accept=getattr(args, "accept", False),
    )


def _check_unresolved_flags(
    workdir: Path,
    ignore: bool = False,
    chapters: list[int] | None = None,
) -> None:
    """raise if review/audit.json contains unresolved flags, unless ignored.

    when `chapters` is given, only flags for those chapters are considered.
    """
    audit_path = get_command_dir(workdir, "review") / "audit.json"
    entries = _load_audit(audit_path)
    flags = [e for e in entries if e.get("kind") == "flag"]
    if chapters is not None:
        wanted = set(chapters)

        def _ch_num(entry: dict) -> int | None:
            stem = entry.get("chapter") or ""
            try:
                return int(stem.split("_")[0])
            except ValueError:
                return None

        flags = [f for f in flags if _ch_num(f) in wanted]
    if not flags:
        return
    if ignore:
        print(f"perform: ignoring {len(flags)} unresolved flag(s) at {audit_path}")
        return
    raise RuntimeError(
        f"{len(flags)} unresolved review flag(s) at {audit_path}. "
        f"run `autiobook audit {workdir}` to resolve, or pass --ignore-flags."
    )


def cmd_perform(args):
    from .utils import get_clone_config

    workdir = Path(args.workdir)
    _check_unresolved_flags(workdir, ignore=getattr(args, "ignore_flags", False))

    chapters = get_chapters(args)
    config = get_clone_config(args)

    run_performance(
        workdir,
        chapters,
        config,
        args.pooled,
        verbose=args.verbose,
        force=args.force,
        retake=getattr(args, "retake", False),
        accept=getattr(args, "accept", False),
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
        accept=getattr(args, "accept", False),
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
    accept: bool = False,
) -> None:
    """fix script issues by filling missing segments and removing hallucinated ones.

    accept=True skips validation/repair entirely — the on-disk scripts are
    trusted as-is.
    """
    if accept:
        print("revise: skipped (--accept)")
        return
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

    summary_parts = []
    if fix_missing and total_added > 0:
        summary_parts.append(f"added {total_added} segment(s)")
    if fix_hallucinated and total_removed > 0:
        summary_parts.append(f"removed {total_removed} segment(s)")

    if summary_parts:
        print(f"\nrevise: {', '.join(summary_parts)}")
    else:
        print("revise: no issues found.")


_WS_RE = re.compile(r"\s+")


def _locate_span(source: str, batch_texts: list[str], cursor: int) -> tuple[int, int]:
    """find the source-text span covering the batch's texts starting at cursor.

    returns (start, end) in source coordinates. walks each batch text forward
    from the cursor; on a miss, falls back to searching the normalized source.
    if nothing locates, the span degenerates to the cursor → end-of-source.
    """
    start = cursor
    end = cursor
    cur = cursor
    for t in batch_texts:
        needle = t.strip()
        if not needle:
            continue
        idx = source.find(needle, cur)
        if idx < 0:
            # fallback: try normalized whitespace match
            norm_needle = _WS_RE.sub(" ", needle)
            norm_tail = _WS_RE.sub(" ", source[cur:])
            ni = norm_tail.find(norm_needle)
            if ni < 0:
                continue
            # approximate: advance cursor by the length of the normalized match
            idx = cur + ni
        if start == cursor and idx >= cursor:
            start = idx
        cur = idx + len(needle)
        end = cur
    if end <= start:
        end = len(source)
    return start, end


def _save_audit(audit_path: Path, entries: list[dict]) -> None:
    """atomically write the review audit log to disk."""
    import json as _json

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = audit_path.with_suffix(audit_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(entries, f, indent=2, ensure_ascii=False)
    tmp.replace(audit_path)


def _load_audit(audit_path: Path) -> list[dict]:
    """load review audit log, returning empty list if missing or malformed."""
    import json as _json

    if not audit_path.exists():
        return []
    try:
        with open(audit_path, encoding="utf-8") as f:
            data = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _edit_script_and_validate(workdir: Path, chapter_stem: str) -> None:
    """launch $EDITOR on the chapter script, then validate against source on save."""
    import os
    import shlex
    import subprocess

    script_path = get_command_dir(workdir, "script") / (chapter_stem + SCRIPT_EXT)
    txt_path = get_command_dir(workdir, "extract") / (chapter_stem + TXT_EXT)
    if not script_path.exists():
        print(f"  edit: script not found: {script_path}")
        return

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    cmd = shlex.split(editor) + [str(script_path)]
    try:
        subprocess.call(cmd)
    except OSError as e:
        print(f"  edit: failed to launch editor ({e})")
        return

    if not txt_path.exists():
        print(f"  validate: source not found: {txt_path}; skipping validation")
        return
    result = validate_script(txt_path, script_path)
    if result.missing or result.hallucinated:
        print(
            f"  validate: FAIL ({len(result.missing)} missing, "
            f"{len(result.hallucinated)} hallucinated)"
        )
        for text, _ins_idx, _off, _line in result.missing[:5]:
            print(f"    missing: {_truncate(text)}")
        if len(result.missing) > 5:
            print(f"    ... +{len(result.missing) - 5} more")
        if result.hallucinated:
            print(f"    hallucinated seg idx: {result.hallucinated[:10]}")
    else:
        print("  validate: OK")


def _print_audit_entry(idx: int, total: int, e: dict) -> None:
    """render a single audit entry with source context for the human reviewer."""
    kind = e.get("kind", "flag")
    print()
    header = f"[{idx + 1}/{total}] ({kind}) {e.get('chapter')} seg {e.get('segment')}"
    print(header)
    if kind == "flag":
        print(f"  reason: {e.get('reason', '')}")
    elif kind == "edit":
        print(
            f"  {e.get('field', '?')}: "
            f"{e.get('before', '')!r} -> {e.get('after', '')!r}"
        )
    elif kind == "validation":
        print(
            f"  batch {e.get('batch', '?')} rejected: "
            f"{e.get('missing', 0)} missing, "
            f"{e.get('hallucinated', 0)} hallucinated"
        )
        for t in e.get("missing_text", [])[:3]:
            print(f"    missing: {_truncate(t)}")
    text = e.get("text", "")
    if text:
        print(f"  text: {text}")
    span = e.get("source_span", "")
    if span:
        print("  --- source context ---")
        for line in span.splitlines():
            print(f"  | {line}")
        print("  --- end source context ---")


def cmd_audit(args) -> None:
    """walk through the review audit log (flags by default; edits with --all)."""
    workdir = Path(args.workdir)
    audit_path = get_command_dir(workdir, "review") / "audit.json"
    entries = _load_audit(audit_path)
    if not entries:
        print(f"audit: no entries at {audit_path}")
        return

    if args.clear:
        audit_path.unlink(missing_ok=True)
        print(f"audit: cleared {len(entries)} entry(s) from {audit_path}")
        return

    # default: flags + validation rejections. --all: include edit records too.
    visible = (
        entries
        if args.all
        else [e for e in entries if e.get("kind") in ("flag", "validation")]
    )
    if not visible:
        print(f"audit: no flags at {audit_path} (use --all to show edits)")
        return

    if args.list:
        for e in visible:
            kind = e.get("kind", "flag")
            head = f"{e.get('chapter', '?')} seg {e.get('segment', '?')} ({kind})"
            if kind == "flag":
                print(f"{head}: {e.get('reason', '')}")
            elif kind == "validation":
                print(
                    f"{head}: batch {e.get('batch', '?')} rejected "
                    f"({e.get('missing', 0)} missing, "
                    f"{e.get('hallucinated', 0)} hallucinated)"
                )
            else:
                print(
                    f"{head}: {e.get('field', '?')} "
                    f"{e.get('before', '')!r} -> {e.get('after', '')!r}"
                )
        return

    idx = 0
    while idx < len(visible):
        e = visible[idx]
        _print_audit_entry(idx, len(visible), e)
        choice = (
            input("  [k]eep / [e]dit / [d]ismiss / [n]ext / [q]uit> ").strip().lower()
        )
        if choice in ("q", "quit"):
            break
        if choice in ("e", "edit"):
            chapter = e.get("chapter", "")
            if chapter:
                _edit_script_and_validate(workdir, chapter)
            else:
                print("  edit: entry has no chapter; skipping")
            continue
        if choice in ("d", "dismiss"):
            entries.remove(e)
            visible.pop(idx)
            _save_audit(audit_path, entries)
            print("  dismissed.")
            continue
        idx += 1

    flags_left = sum(1 for e in entries if e.get("kind") == "flag")
    validation_left = sum(1 for e in entries if e.get("kind") == "validation")
    edits_left = len(entries) - flags_left - validation_left
    print(
        f"audit: {flags_left} flag(s), {validation_left} validation(s), "
        f"{edits_left} edit(s) remaining at {audit_path}"
    )


def run_review(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    batch_size: int = REVIEW_BATCH_SIZE,
    verbose: bool = False,
    force: bool = False,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    accept: bool = False,
) -> None:
    """review each chapter's script in fixed-size batches against the source text.

    each batch is sent to the LLM with its covering source span; the corrected
    segments replace the original batch. per-chapter resume via script-content hash.

    accept=True skips llm review; existing scripts are re-stamped as fresh under
    the current (source + cast + batch_size) hash.
    """
    cast = load_cast(workdir)
    extract_dir = get_command_dir(workdir, "extract")
    script_dir = get_command_dir(workdir, "script")
    txt_files = sorted(extract_dir.glob(f"*{TXT_EXT}"))
    if not txt_files:
        msg = "review: no text files found in extract/!"
        print(msg)
        raise RuntimeError(msg)

    resume = ResumeManager.for_command(workdir, "review", force=force)

    if accept:
        accepted = 0
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
            source = txt_path.read_text(encoding="utf-8")
            task_hash = compute_hash(
                {
                    "source": source,
                    "cast": [(c.name, tuple(c.aliases or ())) for c in cast],
                    "batch_size": batch_size,
                }
            )
            resume.clear_partial(str(num))
            resume.update(str(num), task_hash)
            accepted += 1
        resume.save()
        print(f"review: accepted existing review for {accepted} chapter(s)")
        return
    total_chapters = 0
    total_batches = 0
    audit: list[dict] = []
    audit_path = get_command_dir(workdir, "review") / "audit.json"

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

        segments = load_script(script_path)
        if not segments:
            continue

        source = txt_path.read_text(encoding="utf-8")
        # hash excludes script contents so partial writes to disk don't
        # invalidate the resume record. script identity is on disk; source,
        # cast, and batch_size are the real inputs that gate re-review.
        task_hash = compute_hash(
            {
                "source": source,
                "cast": [(c.name, tuple(c.aliases or ())) for c in cast],
                "batch_size": batch_size,
            }
        )
        if resume.is_fresh(str(num), task_hash):
            continue

        partial = resume.get_partial(str(num))
        if partial and partial.get("hash") == task_hash:
            start_batch = int(partial.get("batches_done", 0))
            cursor = int(partial.get("cursor", 0))
            new_segments = list(segments[: start_batch * batch_size])
            print(
                f"\n{txt_path.name}: resuming review at batch {start_batch + 1} "
                f"({len(segments)} segments, batches of {batch_size})..."
            )
        else:
            start_batch = 0
            cursor = 0
            new_segments = []
            print(
                f"\n{txt_path.name}: reviewing {len(segments)} segments "
                f"in batches of {batch_size}..."
            )

        total_for_chapter = (len(segments) + batch_size - 1) // batch_size
        bar = tqdm(
            total=total_for_chapter,
            initial=start_batch,
            desc=f"review ch{num}",
            unit="batch",
        )
        for i in range(start_batch * batch_size, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            bnum = i // batch_size + 1
            start, end = _locate_span(source, [s.text for s in batch], cursor)
            span = source[start:end] if end > start else source[cursor:]
            batch_flags: list = []
            try:
                corrected, batch_flags = review_script_batch(
                    span,
                    batch,
                    cast,
                    api_base=api_base,
                    api_key=api_key,
                    model=model or DEFAULT_LLM_MODEL,
                    thinking_budget=thinking_budget,
                )
            except Exception as e:
                tqdm.write(f"  batch {bnum}: review failed ({e}); keeping original")
                corrected = batch
            else:
                # validate corrected batch covers the same source span as the
                # original — reject any result that introduces missing fragments
                # or hallucinations relative to this batch's source window.
                result = _validate_segments(span, corrected)
                if result.missing or result.hallucinated:
                    tqdm.write(
                        f"  batch {bnum}: review rejected "
                        f"({len(result.missing)} missing, "
                        f"{len(result.hallucinated)} hallucinated); keeping original"
                    )
                    audit.append(
                        {
                            "kind": "validation",
                            "chapter": txt_path.stem,
                            "segment": i + 1,
                            "batch": bnum,
                            "missing": len(result.missing),
                            "hallucinated": len(result.hallucinated),
                            "missing_text": [t for t, *_ in result.missing[:5]],
                            "source_span": span,
                        }
                    )
                    corrected = batch
            # record accepted edits (speaker/instruction changes) to the audit log.
            for idx, (a, b) in enumerate(zip(batch, corrected)):
                seg_no = i + idx + 1
                for field, before, after in (
                    ("speaker", a.speaker, b.speaker),
                    ("instruction", a.instruction, b.instruction),
                ):
                    if before != after:
                        audit.append(
                            {
                                "kind": "edit",
                                "chapter": txt_path.stem,
                                "segment": seg_no,
                                "field": field,
                                "before": before,
                                "after": after,
                                "text": a.text,
                                "source_span": span,
                            }
                        )
            # record LLM-emitted human-review flags.
            for f in batch_flags:
                seg_no = i + f.index + 1
                audit.append(
                    {
                        "kind": "flag",
                        "chapter": txt_path.stem,
                        "segment": seg_no,
                        "reason": f.reason,
                        "text": batch[f.index].text if f.index < len(batch) else "",
                        "source_span": span,
                    }
                )
                tqdm.write(f"  flag: {txt_path.stem} seg {seg_no}: {f.reason}")
            _save_audit(audit_path, audit)
            new_segments.extend(corrected)
            cursor = end
            total_batches += 1
            # persist progress after each batch so cancel + rerun resumes here.
            save_script(script_path, new_segments + segments[i + batch_size :])
            resume.set_partial(
                str(num),
                {
                    "hash": task_hash,
                    "batches_done": bnum,
                    "cursor": cursor,
                },
            )
            resume.save()
            if verbose:
                changed = [
                    (idx, a, b)
                    for idx, (a, b) in enumerate(zip(batch, corrected))
                    if a.speaker != b.speaker
                    or a.text != b.text
                    or a.instruction != b.instruction
                ]
                if changed:
                    tqdm.write(f"  batch {bnum}: {len(changed)} change(s)")
                    for idx, a, b in changed:
                        seg_no = i + idx + 1
                        if a.speaker != b.speaker:
                            tqdm.write(
                                f"    seg {seg_no} speaker: "
                                f"{a.speaker!r} -> {b.speaker!r}"
                            )
                        if a.instruction != b.instruction:
                            tqdm.write(
                                f"    seg {seg_no} instruction: "
                                f"{a.instruction!r} -> {b.instruction!r}"
                            )
                        if a.text != b.text:
                            tqdm.write(f"    seg {seg_no} text: {a.text!r}")
                            tqdm.write(f"      ->   {b.text!r}")
            bar.update(1)
        bar.close()

        save_script(script_path, new_segments)
        resume.update(str(num), task_hash)
        resume.save()
        total_chapters += 1

    if audit:
        flags_count = sum(1 for e in audit if e.get("kind") == "flag")
        edits_count = sum(1 for e in audit if e.get("kind") == "edit")
        validation_count = sum(1 for e in audit if e.get("kind") == "validation")
        print(
            f"review: {flags_count} flag(s), {edits_count} edit(s), "
            f"{validation_count} validation(s) written to "
            f"{audit_path}; run `autiobook audit <workdir>` to inspect"
        )

    if total_chapters == 0:
        print("review: all chapters up to date.")
    else:
        print(
            f"\nreview: processed {total_chapters} chapter(s), {total_batches} batch(es)"
        )


def cmd_review(args):
    """review scripts against source text in batches (correct speakers, text, emotion)."""
    from .utils import Logger

    chapters = get_chapters(args)
    workdir = Path(args.workdir)
    Logger.init(workdir)

    run_review(
        workdir,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
        batch_size=getattr(args, "batch_size", None) or REVIEW_BATCH_SIZE,
        verbose=args.verbose,
        force=args.force,
        thinking_budget=args.thinking_budget,
        accept=getattr(args, "accept", False),
    )


def _step_if_changed(step: bool, phase: str, path: Path, before: float) -> None:
    """raise StepComplete if step mode is active and files changed."""
    from .utils import dir_mtime

    if step and dir_mtime(path) > before:
        from .main import StepComplete

        raise StepComplete(phase)


def _enumerate_chapters(workdir: Path, chapters: list[int] | None) -> list[int]:
    """list chapter numbers from extracted txt files, filtered by `chapters`."""
    extract_dir = get_command_dir(workdir, "extract")
    nums = []
    for p in sorted(extract_dir.glob(f"*{TXT_EXT}")):
        try:
            n = int(p.stem.split("_")[0])
        except ValueError:
            continue
        if chapters and n not in chapters:
            continue
        nums.append(n)
    return nums


def _run_script_phases(
    workdir: Path,
    chapters: list[int] | None,
    *,
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    verbose: bool,
    force: bool,
    thinking_budget: int,
    revise: bool,
    review: bool,
    step: bool,
    redo_phase: str | None,
    accept: bool = False,
) -> None:
    """run script → revise → [review] for the given chapters.

    review deliberately does not check for audit flags — unresolved flags only
    block perform, not further review passes.
    """
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
        accept=accept,
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
        accept=accept,
    )
    _step_if_changed(step, "revise", get_command_dir(workdir, "script"), before)

    if review:
        before = dir_mtime(get_command_dir(workdir, "script"))
        run_review(
            workdir,
            api_base=api_base,
            api_key=api_key,
            model=model,
            chapters=chapters,
            verbose=verbose,
            force=force or redo_phase == "review",
            thinking_budget=thinking_budget,
            accept=accept,
        )
        _step_if_changed(step, "review", get_command_dir(workdir, "script"), before)


def _run_perform_phases(
    workdir: Path,
    chapters: list[int] | None,
    *,
    clone_config: Any,
    pooled: bool,
    verbose: bool,
    force: bool,
    redo_phase: str | None,
    retake: bool,
    step: bool,
    accept: bool = False,
) -> None:
    """run perform → retake for the given chapters."""
    before = dir_mtime(get_command_dir(workdir, "perform"))
    run_performance(
        workdir,
        chapters,
        clone_config,
        pooled,
        verbose=verbose,
        force=force or redo_phase == "perform",
        retake=retake,
        accept=accept,
    )
    _step_if_changed(step, "perform", get_command_dir(workdir, "perform"), before)

    before = dir_mtime(get_command_dir(workdir, "perform") / "segments")
    from .retake import run_retake

    if accept:
        print("retake: skipped (--accept)")
    else:
        run_retake(
            workdir,
            command="perform",
            chapters=chapters,
            config=clone_config,
            verbose=verbose,
        )
    if not accept:
        _step_if_changed(
            step, "retake", get_command_dir(workdir, "perform") / "segments", before
        )


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
    review: bool = False,
    step: bool = False,
    redo_phase: str | None = None,
    retake: bool = False,
    callback: bool = False,
    emotions: bool = False,
    preset_voices: bool = False,
    directed: bool = False,
    accept: bool = False,
    ignore_flags: bool = False,
    phase_wise: bool = False,
    export_fn: Any = None,
) -> None:
    """run full dramatization pipeline.

    by default every phase (cast → audition → [emote] → script → revise →
    [review] → perform → retake → export) runs chapter-wise: all phases
    complete for chapter 1 before chapter 2 begins. pass phase_wise=True to
    instead run each phase across all chapters before advancing.
    """
    from .audition import run_audition

    audition_config = design_config
    if preset_voices:
        from .config import DEFAULT_MODEL
        from .tts_http import HTTPTTSConfig

        audition_config = HTTPTTSConfig(
            api_base=api_base or "",
            model=DEFAULT_MODEL,
        )
    emote_config = audition_config if preset_voices else design_config

    def _run_head(chs: list[int] | None) -> None:
        before = dir_mtime(get_command_dir(workdir, "cast"))
        run_cast_generation(
            workdir,
            api_base,
            api_key,
            model,
            chs,
            verbose=verbose,
            force=force or redo_phase == "cast",
            thinking_budget=thinking_budget,
            accept=accept,
        )
        _step_if_changed(step, "cast", get_command_dir(workdir, "cast"), before)

        before = dir_mtime(get_command_dir(workdir, "audition"))
        run_audition(
            workdir,
            verbose=verbose,
            force=force or redo_phase == "audition",
            config=audition_config,
            callback=callback,
            preset_voices=preset_voices,
            directed=directed,
            accept=accept,
        )
        _step_if_changed(step, "audition", get_command_dir(workdir, "audition"), before)

        if emotions:
            before = dir_mtime(get_command_dir(workdir, "emote"))
            run_emotes(
                workdir,
                verbose=verbose,
                force=force or redo_phase == "emote",
                config=emote_config,
                callback=callback,
                preset_voices=preset_voices,
                accept=accept,
            )
            _step_if_changed(step, "emote", get_command_dir(workdir, "emote"), before)

    script_kwargs: dict[str, Any] = dict(
        api_base=api_base,
        api_key=api_key,
        model=model,
        verbose=verbose,
        force=force,
        thinking_budget=thinking_budget,
        revise=revise,
        accept=accept,
        review=review,
        step=step,
        redo_phase=redo_phase,
    )
    perform_kwargs: dict[str, Any] = dict(
        clone_config=clone_config,
        pooled=pooled,
        verbose=verbose,
        force=force,
        redo_phase=redo_phase,
        retake=retake,
        step=step,
        accept=accept,
    )

    if phase_wise:
        _run_head(chapters)
        _run_script_phases(workdir, chapters, **script_kwargs)
        _check_unresolved_flags(workdir, ignore=ignore_flags)
        _run_perform_phases(workdir, chapters, **perform_kwargs)
        return

    chapter_nums = _enumerate_chapters(workdir, chapters)
    if not chapter_nums:
        _run_head(chapters)
        _run_script_phases(workdir, chapters, **script_kwargs)
        _check_unresolved_flags(workdir, ignore=ignore_flags)
        _run_perform_phases(workdir, chapters, **perform_kwargs)
        return

    # single chapter-wise pass: everything for chapter N completes (including
    # export) before chapter N+1 begins. unresolved flags from chapter N skip
    # perform/export for that chapter but do not block chapter N+1's review.
    deferred: list[int] = []
    for num in chapter_nums:
        print(f"dramatize: chapter {num}")
        _run_head([num])
        _run_script_phases(workdir, [num], **script_kwargs)
        try:
            _check_unresolved_flags(workdir, ignore=ignore_flags, chapters=[num])
        except RuntimeError as e:
            print(f"dramatize: chapter {num} deferred - {e}")
            deferred.append(num)
            continue
        _run_perform_phases(workdir, [num], **perform_kwargs)
        if export_fn is not None:
            export_dir = workdir / "export"
            before = dir_mtime(export_dir)
            export_fn([num])
            _step_if_changed(step, "export", export_dir, before)

    if deferred:
        raise RuntimeError(
            f"{len(deferred)} chapter(s) deferred due to unresolved flags: "
            f"{deferred}. run `autiobook audit` to resolve, or pass "
            f"--ignore-flags."
        )
