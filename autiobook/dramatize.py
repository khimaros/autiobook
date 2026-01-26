"dramatization workflow logic."

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import soundfile as sf  # type: ignore
from tqdm import tqdm  # type: ignore

from .audio import (
    get_segments_dir,
)
from .config import (
    BASE_MODEL,
    CAST_FILE,
    DEFAULT_CAST,
    SCRIPT_EXT,
    TXT_EXT,
    SAMPLE_RATE,
    VOICE_DESIGN_MODEL,
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
from .tts import (
    TTSConfig,
    TTSEngine,
    chunk_text,
)
from .utils import get_chapters, get_tts_config


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
        updates = []
        new_aliases = set(existing.aliases or [])
        if c.aliases:
            new_aliases.update(c.aliases)
        if merge_source:
            new_aliases.add(merge_source)
            new_aliases.discard(existing.name)

        sorted_aliases = sorted(new_aliases) if new_aliases else None
        if sorted_aliases != existing.aliases:
            existing.aliases = sorted_aliases
            updates.append("aliases")

        if c.description and c.description != existing.description:
            existing.description = c.description
            updates.append("description")

        if verbose and updates:
            msg = f"  {'merged' if merge_source else 'updated'} '{existing.name}'"
            print(f"{msg} ({', '.join(updates)})")

        return "merged" if merge_source else ("updated" if updates else "unchanged")

    # new character
    if verbose:
        print(f"  added new character: '{c.name}'")
    cast_map[c.name.lower()] = c
    if c.aliases:
        for alias in c.aliases:
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
) -> int:
    """process a single batch of chapters for cast generation."""
    full_sample = ""
    for num in batch_chapters:
        txt_path = chapter_map[num]
        full_sample += f"\n--- Chapter {txt_path.stem} ---\n"
        full_sample += txt_path.read_text(encoding="utf-8")[:2000]

    # format current cast for context
    current_cast = list(cast_map.values())
    summary = "\n".join(
        f"- {c.name}: {c.description}"
        + (f" (also known as: {', '.join(c.aliases)})" if c.aliases else "")
        for c in current_cast
    )

    batch_cast = generate_cast(
        full_sample, api_base, api_key, model or "gpt-4o", existing_cast_summary=summary
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

    batch_size = 3
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


def run_auditions(
    workdir: Path,
    cast: List[Character] | None = None,
    verbose: bool = False,
    force: bool = False,
) -> None:
    """generate voice samples for cast."""

    if cast is None:
        cast = load_cast(workdir)

    voices_dir = get_command_dir(workdir, "audition")
    resume = ResumeManager.for_command(workdir, "audition", force=force)

    if not cast:
        cast_path = get_command_dir(workdir, "cast") / CAST_FILE
        if cast_path.exists():
            print(f"cast file found at {cast_path} but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return

    if len(cast) <= 3 and cast[0].name == "Narrator":
        print(
            "warning: using default cast (Narrator + Extras). run 'cast' to generate full cast."
        )

    engine = TTSEngine(TTSConfig(model_name=VOICE_DESIGN_MODEL))

    print(f"generating auditions for {len(cast)} characters...")

    generated_count = 0
    skipped_count = 0

    for char in tqdm(cast, desc="casting voices"):
        wav_path = voices_dir / f"{char.name}{WAV_EXT}"

        # input data for this character's voice
        char_data = {
            "name": char.name,
            "description": char.description,
            "audition_line": char.audition_line,
        }
        char_hash = compute_hash(char_data)

        if not force and wav_path.exists() and resume.is_fresh(char.name, char_hash):
            skipped_count += 1
            if verbose:
                tqdm.write(f"  skipping {char.name} (up to date)")
            continue

        if verbose:
            tqdm.write(f"  generating {char.name}: '{char.audition_line}'")

        try:
            audio, sr = engine.design_voice(
                text=char.audition_line, instruct=char.description
            )
            sf.write(str(wav_path), audio, sr)
            resume.update(char.name, char_hash)
            resume.save()
            generated_count += 1
        except Exception as e:
            print(f"failed to generate voice for {char.name}: {e}")

    resume.save()
    if generated_count == 0 and skipped_count > 0:
        print("audition: all voices up to date.")
    else:
        print(f"audition: {generated_count} generated, {skipped_count} skipped")


def run_script_generation(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    verbose: bool = False,
    force: bool = False,
) -> bool:
    """generate dramatized scripts for chapters incrementally."""

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
            print("cast file found but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return False

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
            chunk_text = chunks[j]
            try:
                chunk_segments = process_script_chunk(
                    chunk_text, cast, api_base, api_key, model or "gpt-4o"
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
                print(f"\n    chunk {j + 1} FAILED: {e}")
                return False

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
    config: TTSConfig | None = None,
    pooled: bool = False,
    verbose: bool = False,
    force: bool = False,
) -> None:
    """synthesize audio from scripts with segment-level resume."""

    cast = load_cast(workdir)
    if not cast:
        if (get_command_dir(workdir, "cast") / CAST_FILE).exists():
            print("cast file found but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return

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
        print("no voices found. run 'audition' command first.")
        return

    # use provided config but override model to BASE_MODEL for voice cloning
    if config is None:
        config = TTSConfig(model_name=BASE_MODEL)
    else:
        config = TTSConfig(
            model_name=BASE_MODEL,
            batch_size=config.batch_size,
            chunk_size=config.chunk_size,
            compile_model=config.compile_model,
            warmup=config.warmup,
            do_sample=config.do_sample,
            temperature=config.temperature,
        )
    engine = TTSEngine(config)

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
        print("perform: no scripts found.")
        return

    # resume manager for assembly
    resume = ResumeManager.for_command(workdir, "perform", force=force)

    # always use pooled strategy for best performance/caching
    _perform_pooled(engine, pending, voices_dir, cast_map, resume=resume, force=force)


def _perform_pooled(
    engine: TTSEngine,
    pending: list[tuple[Path, Path]],
    voices_dir: Path,
    cast_map: dict[str, Character],
    resume: ResumeManager | None = None,
    force: bool = False,
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

    chapter_data = []
    segments_dir = get_segments_dir(pending[0][1].parent)

    for txt_path, wav_path in pending:
        segments = load_script(txt_path)
        if not segments:
            continue

        chapter_tasks = []
        for segment in segments:
            char_opt = cast_map.get(segment.speaker) or cast_map.get("Narrator")
            char_name = char_opt.name if char_opt else ""
            char_hash = char_hashes.get(char_name, "")

            ref_audio_path = (
                voices_dir / f"{char_opt.name}{WAV_EXT}" if char_opt else None
            )
            ref_text = char_opt.audition_line if char_opt else None

            seg_data = {
                "text": segment.text,
                "speaker": segment.speaker,
                "instruction": segment.instruction,
                "char_hash": char_hash,
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
                        instruct=segment.instruction
                        or (char_opt.description if char_opt else ""),
                    )
                )
        chapter_data.append((wav_path, chapter_tasks))

    process_audio_pipeline(
        engine, chapter_data, resume=resume, desc="performing segments", force=force
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
    )


def cmd_audition(args):
    run_auditions(
        Path(args.workdir),
        verbose=args.verbose,
        force=args.force,
    )


def cmd_script(args):
    chapters = get_chapters(args)

    run_script_generation(
        Path(args.workdir),
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
        verbose=args.verbose,
        force=args.force,
    )


def cmd_perform(args):
    chapters = get_chapters(args)
    config = get_tts_config(args)

    run_performance(
        Path(args.workdir),
        chapters,
        config,
        args.pooled,
        verbose=args.verbose,
        force=args.force,
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

    missing: list[tuple[str, int, int]]  # (text, insertion_index, split_offset)
    hallucinated: list[int]  # indices of segments not found in source


def validate_script(txt_path: Path, script_path: Path) -> ValidationResult:
    """validate that script segments match the source text using fuzzy diffing.

    uses difflib to align normalized words between source and script.
    identifies missing text (source words not in script) and hallucinated
    segments (segments with low word match ratio).
    """
    segments = load_script(script_path)
    if not segments:
        return ValidationResult(
            missing=[(f"no script found for {txt_path.name}", 0, 0)], hallucinated=[]
        )

    original_text = txt_path.read_text(encoding="utf-8")

    # 1. Tokenize source and script
    source_tokens = _tokenize_with_positions(original_text)
    source_words = [t[0] for t in source_tokens]

    script_words = []
    script_token_info = []  # (seg_idx, start, end)

    segment_stats = {}  # seg_idx -> {'total': 0, 'matched': 0}

    for i, seg in enumerate(segments):
        # use the same tokenizer for segments
        seg_tokens = _tokenize_with_positions(seg.text)
        segment_stats[i] = {"total": len(seg_tokens), "matched": 0}
        for t in seg_tokens:
            script_words.append(t[0])
            script_token_info.append((i, t[1], t[2]))

    # 2. Diff
    matcher = difflib.SequenceMatcher(None, source_words, script_words, autojunk=False)
    opcodes = matcher.get_opcodes()

    missing_ranges = []  # list of (start_char, end_char, insertion_index, split_offset)

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            # record matches for hallucination detection
            for j in range(j1, j2):
                seg_idx = script_token_info[j][0]
                segment_stats[seg_idx]["matched"] += 1
        elif tag in ("delete", "replace"):
            # source words i1:i2 are missing (or replaced)
            if i1 < i2:
                start_char = source_tokens[i1][1]
                end_char = source_tokens[i2 - 1][2]

                # determine insertion index based on script position j1
                if j1 < len(script_words):
                    ins_idx, split_offset, _ = script_token_info[j1]
                else:
                    ins_idx = len(segments)
                    split_offset = 0

                missing_ranges.append((start_char, end_char, ins_idx, split_offset))

    # 3. Merge contiguous missing ranges
    missing_fragments = []
    if missing_ranges:
        missing_ranges.sort()
        merged = [missing_ranges[0]]

        for current_start, current_end, current_ins, current_offset in missing_ranges[
            1:
        ]:
            last_start, last_end, last_ins, last_offset = merged[-1]

            # check gap between last_end and current_start
            gap_text = original_text[last_end:current_start]

            # merge if:
            # 1. Same insertion index
            # 2. Same split offset (or very close/sequential?)
            #    Actually, if we merge, we keep the FIRST insertion point (last_ins, last_offset).
            #    But we should only merge if the gap is just punctuation/whitespace.
            #    And typically they will be at the same insertion point if they are
            #    contiguous in source but skipped in script.
            if (
                current_ins == last_ins
                and current_offset == last_offset
                and not re.search(r"\w", gap_text)
            ):
                merged[-1] = (last_start, current_end, last_ins, last_offset)
            else:
                merged.append((current_start, current_end, current_ins, current_offset))

        for start, end, ins_idx, split_offset in merged:
            # expand to include adjacent punctuation but not whitespace
            while start > 0 and original_text[start - 1] in ".,;:?!\"'()[]-":
                start -= 1
            while end < len(original_text) and original_text[end] in ".,;:?!\"'()[]-":
                end += 1

            text = original_text[start:end].strip()
            # filter out tiny fragments
            if len(text) > 1 or (len(text) == 1 and text.isalnum()):
                missing_fragments.append((text, ins_idx, split_offset))

    # 4. Identify hallucinated segments
    hallucinated_indices = []
    for i in range(len(segments)):
        stats = segment_stats[i]
        if stats["total"] == 0:
            continue

        ratio = stats["matched"] / stats["total"]
        if ratio < 0.5:  # less than 50% words matched
            hallucinated_indices.append(i)

    return ValidationResult(
        missing=missing_fragments, hallucinated=hallucinated_indices
    )


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

            if check_missing:
                for i, (fragment, idx, offset) in enumerate(result.missing, 1):
                    print(f"  [missing {i} @ {idx}+{offset}] {fragment}")

            if check_hallucinated:
                segments = load_script(script_path)
                for idx in result.hallucinated:
                    seg = segments[idx]
                    print(f"  [hallucinated {idx}] {seg.speaker}: {seg.text}")
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
        print(
            f"\nvalidate: found {', '.join(summary_parts)} across {len(results)} chapters"
        )

    return results


def cmd_validate(args):
    chapters = get_chapters(args)

    # default to checking both if neither flag specified
    check_missing = args.missing or (not args.missing and not args.hallucinated)
    check_hallucinated = args.hallucinated or (
        not args.missing and not args.hallucinated
    )

    run_validation(Path(args.workdir), chapters, check_missing, check_hallucinated)


DEFAULT_CONTEXT_CHARS = 500  # default characters of context before/after missing text
DEFAULT_CONTEXT_PARAGRAPHS = (
    2  # default paragraphs of context before/after missing text
)


def _extract_context(
    original_text: str,
    fragment: str,
    context_chars: int | None = None,
    context_paragraphs: int | None = None,
) -> tuple[str, str]:
    """extract context before and after a missing fragment.

    if context_paragraphs is set, uses paragraph boundaries.
    otherwise uses context_chars (default 500).

    returns (context_before, context_after) as strings.
    """
    if context_paragraphs is not None:
        return _extract_context_paragraphs(original_text, fragment, context_paragraphs)

    # character-based extraction
    original_norm = _normalize_text(original_text)
    fragment_norm = _normalize_text(fragment)
    chars = context_chars or DEFAULT_CONTEXT_CHARS

    pos = original_norm.find(fragment_norm[:50])
    if pos == -1:
        return "", ""

    start = max(0, pos - chars)
    context_before = original_norm[start:pos].strip()

    end_pos = pos + len(fragment_norm)
    context_after = original_norm[end_pos : end_pos + chars].strip()

    return context_before, context_after


def _extract_context_paragraphs(
    original_text: str, fragment: str, num_paragraphs: int
) -> tuple[str, str]:
    """extract context using paragraph boundaries."""
    paragraphs = original_text.split("\n\n")
    fragment_norm = _normalize_text(fragment)

    # find which paragraph contains the start of the fragment
    target_idx = None
    for i, para in enumerate(paragraphs):
        para_norm = _normalize_text(para)
        if fragment_norm[:50] in para_norm:
            target_idx = i
            break

    if target_idx is None:
        return "", ""

    # extract paragraphs before
    start_idx = max(0, target_idx - num_paragraphs)
    context_before = "\n\n".join(paragraphs[start_idx:target_idx])

    # extract paragraphs after
    end_idx = min(len(paragraphs), target_idx + num_paragraphs + 1)
    context_after = "\n\n".join(paragraphs[target_idx + 1 : end_idx])

    return context_before.strip(), context_after.strip()


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


def _fill_missing_fragments(
    segments: List[ScriptSegment],
    missing: list[tuple[str, int, int]],
    original_text: str,
    cast: List[Character],
    api_base: str | None,
    api_key: str | None,
    model: str | None,
    context_chars: int | None,
    context_paragraphs: int | None,
    verbose: bool,
) -> int:
    """fill missing text fragments using LLM."""
    added = 0
    # Sort descending so insertions don't invalidate subsequent indices
    for fragment, insertion_idx, split_offset in sorted(
        missing, key=lambda x: (x[1], x[2]), reverse=True
    ):
        if verbose:
            print(
                f"\n    missing fragment (@ {insertion_idx}+{split_offset}): {fragment}"
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

        context_before, context_after = _extract_context(
            original_text, fragment, context_chars, context_paragraphs
        )

        try:
            new_segs = fix_missing_segment(
                fragment,
                context_before,
                context_after,
                cast,
                api_base,
                api_key,
                model or "gpt-4o",
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
    return added


def run_fix(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    fix_missing: bool = True,
    fix_hallucinated: bool = True,
    context_chars: int | None = None,
    context_paragraphs: int | None = None,
    verbose: bool = False,
) -> None:
    """fix script issues by filling missing segments and removing hallucinated ones."""
    cast = load_cast(workdir)
    extract_dir, script_dir = (
        get_command_dir(workdir, "extract"),
        get_command_dir(workdir, "script"),
    )
    txt_files = sorted(extract_dir.glob(f"*{TXT_EXT}"))

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
                    context_chars,
                    context_paragraphs,
                    verbose,
                )
                save_script(script_path, segments)

    if total_added > 0 or total_removed > 0:
        print(f"\nfix: added {total_added}, removed {total_removed} segment(s)")
    else:
        print("fix: no issues found.")

    # summary
    summary_parts = []
    if fix_missing and total_added > 0:
        summary_parts.append(f"added {total_added} segment(s)")
    if fix_hallucinated and total_removed > 0:
        summary_parts.append(f"removed {total_removed} segment(s)")

    if summary_parts:
        print(f"\nfix: {', '.join(summary_parts)}")
    else:
        print("fix: no issues found.")


def cmd_fix(args):
    chapters = get_chapters(args)

    # default to fixing both if neither flag specified
    fix_missing = args.missing or (not args.missing and not args.hallucinated)
    fix_hallucinated = args.hallucinated or (not args.missing and not args.hallucinated)

    # context flags are mutually exclusive
    context_chars = getattr(args, "context_chars", None)
    context_paragraphs = getattr(args, "context_paragraphs", None)

    run_fix(
        Path(args.workdir),
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
        fix_missing=fix_missing,
        fix_hallucinated=fix_hallucinated,
        context_chars=context_chars,
        context_paragraphs=context_paragraphs,
        verbose=args.verbose,
    )


def dramatize_book(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    tts_config: TTSConfig | None = None,
    pooled: bool = False,
    verbose: bool = False,
    force: bool = False,
) -> None:
    """run full dramatization pipeline."""
    run_cast_generation(
        workdir, api_base, api_key, model, chapters, verbose=verbose, force=force
    )
    # generate scripts first before auditions
    if not run_script_generation(
        workdir, api_base, api_key, model, chapters, verbose=verbose, force=force
    ):
        print("script generation failed. aborting.")
        return

    # fix script issues (missing/hallucinated)
    run_fix(workdir, api_base, api_key, model, chapters, verbose=verbose)

    # now run auditions
    run_auditions(workdir, verbose=verbose, force=force)
    run_performance(
        workdir,
        chapters,
        tts_config,
        pooled,
        verbose=verbose,
        force=force,
    )
