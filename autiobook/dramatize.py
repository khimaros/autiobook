"dramatization workflow logic."

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .audio import concatenate_audio
from .config import (
    BASE_MODEL,
    CAST_FILE,
    PARAGRAPH_PAUSE_MS,
    SAMPLE_RATE,
    SCRIPT_EXT,
    TXT_EXT,
    VOICE_DESIGN_MODEL,
    WAV_EXT,
    DEFAULT_CAST,
)
from .llm import (
    Character,
    ScriptSegment,
    fix_missing_segment,
    generate_cast,
    process_script_chunk,
    split_text_smart,
)
from .tts import (
    TTSConfig,
    TTSEngine,
    chunk_text,
    finalize_chunks,
    get_chunk_dir,
    load_chunk_progress,
    save_chunk_audio,
    save_chunk_progress,
)
from .utils import iter_pending_chapters


def save_cast(workdir: Path, cast: List[Character], analyzed_chapters: List[int] = None) -> None:
    """save cast to json file."""
    path = workdir / CAST_FILE

    characters = []
    for c in cast:
        char_data = {
            "name": c.name,
            "description": c.description,
            "audition_line": c.audition_line,
        }
        if c.aliases:
            char_data["aliases"] = c.aliases
        if c.appearances > 0:
            char_data["appearances"] = c.appearances
        characters.append(char_data)

    data = {
        "version": 3,
        "analyzed_chapters": analyzed_chapters or [],
        "characters": characters,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cast(workdir: Path) -> tuple[List[Character], List[int]]:
    """load cast from json file. returns (characters, analyzed_chapters)."""
    path = workdir / CAST_FILE
    if not path.exists():
        return [Character(**c) for c in DEFAULT_CAST], []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # handle legacy list format
    if isinstance(data, list):
        chars = []
        for c in data:
            c.setdefault("aliases", None)
            c.setdefault("appearances", 0)
            chars.append(Character(**c))
        return chars, []

    # handle dict format (v1, v2, v3)
    chars = []
    for c in data.get("characters", []):
        c.setdefault("aliases", None)
        c.setdefault("appearances", 0)
        chars.append(Character(**c))
    analyzed = data.get("analyzed_chapters", [])
    return chars, analyzed


def save_script(
    chapter_file: Path,
    segments: List[ScriptSegment],
    completed_chunks: int | None = None,
    total_chunks: int | None = None,
) -> None:
    """save dramatized script for a chapter with progress tracking."""
    script_path = chapter_file.with_suffix(SCRIPT_EXT)
    data = {
        "version": 1,
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
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


def load_script(chapter_file: Path) -> tuple[List[ScriptSegment], int, int]:
    """load dramatized script for a chapter.

    returns (segments, completed_chunks, total_chunks).
    """
    script_path = chapter_file.with_suffix(SCRIPT_EXT)
    if not script_path.exists():
        return [], 0, 0

    with open(script_path, encoding="utf-8") as f:
        data = json.load(f)

    # handle legacy format (list or dict without version)
    if isinstance(data, list):
        segments = [ScriptSegment(**s) for s in data]
        return segments, len(segments), len(segments)

    segments = [ScriptSegment(**s) for s in data.get("segments", [])]
    completed = data.get("completed_chunks", len(segments))
    total = data.get("total_chunks", len(segments))

    return segments, completed, total


def count_appearances(workdir: Path, chapters: list[int] | None = None) -> dict[str, int]:
    """count speaking appearances for each character across scripts."""
    from collections import Counter

    counts = Counter()
    script_files = sorted(workdir.glob(f"*{SCRIPT_EXT}"))

    for script_path in script_files:
        # filter by chapter if specified
        if chapters:
            try:
                chapter_num = int(script_path.stem.split("_")[0])
                if chapter_num not in chapters:
                    continue
            except ValueError:
                continue

        segments, _, _ = load_script(script_path.with_suffix(TXT_EXT))
        for seg in segments:
            counts[seg.speaker] += 1

    return dict(counts)


def _merge_character_into_cast(
    c: Character,
    cast_map: dict[str, Character],
    alias_map: dict[str, str],
    verbose: bool = False,
) -> str:
    """merge a character into the cast, returns 'added', 'updated', or 'merged'."""
    key = c.name.lower()
    existing = None
    merge_source = None  # name of the character being merged into existing (if different)

    # 1. Check if this name is an alias of an existing character
    if key in alias_map:
        canonical_key = alias_map[key]
        existing = cast_map[canonical_key]
        merge_source = c.name

    # 2. Check if any of this character's aliases match an existing character
    elif c.aliases:
        for alias in c.aliases:
            alias_lower = alias.lower()
            if alias_lower in cast_map:
                existing = cast_map[alias_lower]
                merge_source = c.name
                break
            if alias_lower in alias_map:
                existing = cast_map[alias_map[alias_lower]]
                merge_source = c.name
                break

    # 3. Check exact match
    if not existing and key in cast_map:
        existing = cast_map[key]

    if existing:
        updates = []
        changed = False

        # Calculate new aliases
        new_aliases = set(existing.aliases or [])
        if c.aliases:
            new_aliases.update(c.aliases)
        
        if merge_source:
            new_aliases.add(merge_source)
            new_aliases.discard(existing.name)
            
        sorted_aliases = sorted(new_aliases) if new_aliases else None
        
        # Check for alias changes
        if sorted_aliases != existing.aliases:
            if verbose:
                old_aliases = set(existing.aliases or [])
                added = set(sorted_aliases or []) - old_aliases
                if added:
                    updates.append(f"aliases (+{', '.join(added)})")
                else:
                    updates.append("aliases")
            else:
                updates.append("aliases")
            
            existing.aliases = sorted_aliases
            changed = True

        # Check for description changes (always update if different)
        old_desc = existing.description
        if c.description and c.description != old_desc:
            existing.description = c.description
            changed = True
            updates.append("description")

        # Logging
        if verbose:
            if merge_source:
                # Merge message
                msg = f"  merged '{c.name}' into '{existing.name}'"
                if updates:
                    msg += f" ({', '.join(updates)})"
                print(msg)
            elif changed:
                # Update message
                print(f"  updated '{existing.name}': {', '.join(updates)}")

            # Show description diff if it changed
            if "description" in updates:
                if old_desc:
                    print(f"    old description: {old_desc}")
                print(f"    new description: {existing.description}")

        return "merged" if merge_source else ("updated" if changed else "unchanged")

    # new character
    if verbose:
        print(f"  added new character: '{c.name}'")
        if c.description:
            print(f"    description: {c.description}")
    cast_map[key] = c
    if c.aliases:
        for alias in c.aliases:
            alias_map[alias.lower()] = key
    return "added"


def run_cast_generation(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    verbose: bool = False,
) -> List[Character]:
    """analyze book and generate cast list."""
    existing_cast, analyzed_chapters = load_cast(workdir)

    txt_files = sorted(workdir.glob(f"*{TXT_EXT}"))
    if not txt_files:
        print("no extracted text files found!")
        return existing_cast

    # map chapter numbers to files
    chapter_map = {}
    for txt_path in txt_files:
        try:
            num = int(txt_path.stem.split("_")[0])
            chapter_map[num] = txt_path
        except ValueError:
            continue

    all_chapter_nums = sorted(chapter_map.keys())

    # determine which chapters to process
    if chapters:
        chapters_to_process = [c for c in chapters if c in chapter_map]
    else:
        chapters_to_process = [num for num in all_chapter_nums if num not in analyzed_chapters]

    if not chapters_to_process:
        if not chapters:
            print(f"all chapters ({len(all_chapter_nums)}) have been analyzed for cast.")
            return existing_cast
        else:
            print("no matching chapters found to process.")
            return existing_cast

    print(f"analyzing {len(chapters_to_process)} chapters for cast...")

    # build lookup maps from existing cast
    cast_map = {c.name.lower(): c for c in existing_cast}
    alias_map = {}
    for c in existing_cast:
        if c.aliases:
            for alias in c.aliases:
                alias_map[alias.lower()] = c.name.lower()

    # track stats across all batches
    added_count = 0
    updated_count = 0
    merged_count = 0
    total_processed = 0

    # process in batches to avoid overwhelming the LLM
    batch_size = 3
    num_batches = (len(chapters_to_process) + batch_size - 1) // batch_size

    for batch_start in range(0, len(chapters_to_process), batch_size):
        batch_chapters = chapters_to_process[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        # collect samples for this batch
        full_sample = ""
        for num in batch_chapters:
            txt_path = chapter_map[num]
            text = txt_path.read_text(encoding="utf-8")
            full_sample += f"\n--- Chapter {txt_path.stem} ---\n"
            full_sample += text[:2000]

        # format current cast for context (include aliases)
        current_cast = list(cast_map.values())
        summary = ""
        if current_cast:
            lines = []
            for c in current_cast:
                line = f"- {c.name}: {c.description}"
                if c.aliases:
                    line += f" (also known as: {', '.join(c.aliases)})"
                lines.append(line)
            summary = "\n".join(lines)

        print(
            f"  batch {batch_num}/{num_batches}: chapters {batch_chapters} ({len(current_cast)} characters known)..."
        )

        batch_cast = generate_cast(
            full_sample, api_base, api_key, model or "gpt-4o", existing_cast_summary=summary
        )
        total_processed += len(batch_cast)

        # merge batch results into cast
        for c in batch_cast:
            result = _merge_character_into_cast(c, cast_map, alias_map, verbose=verbose)
            if result == "added":
                added_count += 1
            elif result == "updated":
                updated_count += 1
            elif result == "merged":
                merged_count += 1

        # update analyzed chapters and save after each batch
        analyzed_chapters = sorted(list(set(analyzed_chapters + batch_chapters)))
        final_cast = list(cast_map.values())

        # ensure Narrator is at top if present
        narrator = next((c for c in final_cast if c.name.lower() == "narrator"), None)
        if narrator:
            final_cast.remove(narrator)
            final_cast.insert(0, narrator)

        save_cast(workdir, final_cast, analyzed_chapters)

    print(
        f"processed {total_processed} character mentions from {len(chapters_to_process)} chapters."
    )
    print(f"stats: {added_count} added, {updated_count} updated, {merged_count} merged (aliases).")
    print(
        f"final cast: {len(final_cast)} characters. analyzed: {len(analyzed_chapters)}/{len(all_chapter_nums)} chapters."
    )

    return final_cast


def run_auditions(
    workdir: Path,
    cast: List[Character] | None = None,
    min_appearances: int = 0,
    verbose: bool = False,
) -> None:
    """generate voice samples for cast."""
    if cast is None:
        cast, _ = load_cast(workdir)

    if not cast:
        if (workdir / CAST_FILE).exists():
            print(f"cast file found at {workdir / CAST_FILE} but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return

    voices_dir = workdir / "voices"
    voices_dir.mkdir(exist_ok=True)

    # determine which characters need dedicated voices vs generic extras
    main_cast = []
    minor_count = 0
    for char in cast:
        # narrator and characters meeting threshold get dedicated voices
        if char.name.lower() == "narrator" or char.appearances >= min_appearances:
            main_cast.append(char)
        else:
            minor_count += 1

    engine = TTSEngine(TTSConfig(model_name=VOICE_DESIGN_MODEL))

    if min_appearances > 0:
        print(
            f"generating auditions for {len(main_cast)} characters "
            f"({minor_count} minor characters will use generic voices)..."
        )
    else:
        print(f"generating auditions for {len(main_cast)} characters...")

    for char in tqdm(main_cast, desc="casting voices"):
        wav_path = voices_dir / f"{char.name}{WAV_EXT}"
        if wav_path.exists():
            if verbose:
                tqdm.write(f"  skipping {char.name} (exists)")
            continue

        if verbose:
            tqdm.write(f"  generating {char.name}: '{char.audition_line}'")

        try:
            audio, sr = engine.design_voice(text=char.audition_line, instruct=char.description)
            sf.write(str(wav_path), audio, sr)
        except Exception as e:
            print(f"failed to generate voice for {char.name}: {e}")


def run_script_generation(
    workdir: Path,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    chapters: list[int] | None = None,
    verbose: bool = False,
) -> None:
    """generate dramatized scripts for chapters incrementally."""
    cast, _ = load_cast(workdir)
    if not cast:
        if (workdir / CAST_FILE).exists():
            print(f"cast file found at {workdir / CAST_FILE} but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return

    # collect chapters to process
    txt_files = sorted(workdir.glob(f"*{TXT_EXT}"))
    pending_chapters = []
    for txt_path in txt_files:
        try:
            chapter_num = int(txt_path.stem.split("_")[0])
        except ValueError:
            continue
        if chapters and chapter_num not in chapters:
            continue
        pending_chapters.append((chapter_num, txt_path))

    if not pending_chapters:
        print("no chapters to process.")
        return

    # count completed vs pending
    completed_count = 0
    pending_count = 0
    for _, txt_path in pending_chapters:
        _, completed, total = load_script(txt_path)
        text = txt_path.read_text(encoding="utf-8")
        actual_total = len(split_text_smart(text))
        if completed >= actual_total and completed > 0:
            completed_count += 1
        else:
            pending_count += 1

    print(
        f"script generation: {pending_count} chapters to process, {completed_count} already complete"
    )

    total_segments = 0
    chapters_processed = 0

    for chapter_idx, (chapter_num, txt_path) in enumerate(pending_chapters):
        text = txt_path.read_text(encoding="utf-8")
        chunks = split_text_smart(text)
        total_chunks = len(chunks)

        # load existing progress
        existing_segments, completed_chunks, saved_total = load_script(txt_path)

        # if source changed (different chunk count), restart
        if saved_total and saved_total != total_chunks:
            print(
                f"  [{chapter_idx + 1}/{len(pending_chapters)}] restarting {txt_path.name} (source changed)..."
            )
            existing_segments = []
            completed_chunks = 0

        # check if already complete
        if completed_chunks >= total_chunks and existing_segments:
            continue

        if completed_chunks > 0:
            status = f"resuming at chunk {completed_chunks + 1}"
        else:
            status = "starting"

        print(
            f"  [{chapter_idx + 1}/{len(pending_chapters)}] {txt_path.name}: {status} ({total_chunks} chunks)"
        )

        segments = list(existing_segments)

        for i in tqdm(
            range(completed_chunks, total_chunks),
            desc=f"    chapter {chapter_num}",
            unit="chunk",
            initial=completed_chunks,
            total=total_chunks,
        ):
            chunk = chunks[i]
            try:
                chunk_segments = process_script_chunk(
                    chunk, cast, api_base, api_key, model or "gpt-4o"
                )
                if verbose:
                    speakers = set(s.speaker for s in chunk_segments)
                    tqdm.write(
                        f"      chunk {i+1}: generated {len(chunk_segments)} segments. "
                        f"Speakers: {', '.join(sorted(speakers))}"
                    )
                segments.extend(chunk_segments)
                completed_chunks = i + 1
                save_script(txt_path, segments, completed_chunks, total_chunks)

            except Exception as e:
                print(f"\n    chunk {i + 1} FAILED: {e}")
                return

        total_segments += len(segments)
        chapters_processed += 1
        print(f"    -> {len(segments)} segments")

    print(f"done: {chapters_processed} chapters, {total_segments} total segments")

    # update appearance counts in cast
    appearances = count_appearances(workdir, chapters)
    cast, analyzed = load_cast(workdir)
    for c in cast:
        # sum appearances for character and all aliases
        total = appearances.get(c.name, 0)
        if c.aliases:
            for alias in c.aliases:
                total += appearances.get(alias, 0)
        c.appearances = total
    save_cast(workdir, cast, analyzed)
    print(f"updated appearance counts for {len(cast)} characters")


def run_performance(
    workdir: Path,
    chapters: list[int] | None = None,
    config: TTSConfig | None = None,
    pooled: bool = False,
    min_appearances: int = 0,
    verbose: bool = False,
) -> None:
    """synthesize audio from scripts with segment-level resume."""
    cast, _ = load_cast(workdir)
    if not cast:
        if (workdir / CAST_FILE).exists():
            print(f"cast file found at {workdir / CAST_FILE} but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return

    # build cast map including aliases
    # minor characters (below min_appearances) get mapped to generic Extra voices
    cast_map = {}
    for c in cast:
        # narrator always gets own voice
        if c.name.lower() == "narrator" or c.appearances >= min_appearances:
            cast_map[c.name] = c
            if c.aliases:
                for alias in c.aliases:
                    cast_map[alias] = c
        elif min_appearances > 0:
            # map minor character to generic extra (alternate male/female)
            # use description to guess gender, default to male
            desc_lower = c.description.lower()
            is_female = any(w in desc_lower for w in ["female", "woman", "girl", "she", "her"])
            extra_name = "Extra Female" if is_female else "Extra Male"
            extra_char = Character(extra_name, "", "", appearances=0)
            cast_map[c.name] = extra_char
            if c.aliases:
                for alias in c.aliases:
                    cast_map[alias] = extra_char

    voices_dir = workdir / "voices"

    if not voices_dir.exists():
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

    for txt_path, wav_path in iter_pending_chapters(workdir, chapters, skip_message="audio exists"):
        segments, _, _ = load_script(txt_path)
        if not segments:
            print(f"skipping {txt_path.name} (no script found)")
            continue

        _perform_chapter_resumable(engine, txt_path, wav_path, segments, voices_dir, cast_map)


def _perform_chapter_resumable(
    engine: TTSEngine,
    txt_path: Path,
    wav_path: Path,
    segments: List[ScriptSegment],
    voices_dir: Path,
    cast_map: dict[str, Character],
) -> None:
    """synthesize dramatized chapter with segment-level progress tracking."""
    import shutil

    total_segments = len(segments)
    chunk_dir = get_chunk_dir(wav_path)

    # check for existing progress
    completed, saved_total = load_chunk_progress(chunk_dir)

    # restart if source changed
    if saved_total and saved_total != total_segments:
        print(f"restarting {txt_path.name} (script changed)...")
        shutil.rmtree(chunk_dir, ignore_errors=True)
        completed = 0

    # check if already complete
    if completed >= total_segments:
        print(f"finalizing {txt_path.name}...")
        finalize_chunks(chunk_dir, wav_path, total_segments)
        print(f"  -> {wav_path.name}")
        return

    chunk_dir.mkdir(parents=True, exist_ok=True)

    if completed > 0:
        print(f"resuming {txt_path.name} at segment {completed + 1}/{total_segments}...")
    else:
        print(f"synthesizing {txt_path.name} with cast ({total_segments} segments)...")

    for seg_idx in tqdm(
        range(completed, total_segments),
        desc="performing",
        unit="seg",
        initial=completed,
        total=total_segments,
        leave=False,
    ):
        segment = segments[seg_idx]
        character = cast_map.get(segment.speaker) or cast_map.get("Narrator")

        if not character:
            # save silence for missing character
            save_chunk_audio(chunk_dir, seg_idx, np.zeros(1, dtype=np.float32), SAMPLE_RATE)
            save_chunk_progress(chunk_dir, seg_idx + 1, total_segments)
            continue

        ref_audio_path = voices_dir / f"{character.name}{WAV_EXT}"
        if not ref_audio_path.exists():
            save_chunk_audio(chunk_dir, seg_idx, np.zeros(1, dtype=np.float32), SAMPLE_RATE)
            save_chunk_progress(chunk_dir, seg_idx + 1, total_segments)
            continue

        chunks = [c for c in chunk_text(segment.text, engine.config.chunk_size) if c.strip()]
        if not chunks:
            save_chunk_audio(chunk_dir, seg_idx, np.zeros(1, dtype=np.float32), SAMPLE_RATE)
            save_chunk_progress(chunk_dir, seg_idx + 1, total_segments)
            continue

        try:
            # synthesize all chunks for this segment
            segment_audio = []
            for i in range(0, len(chunks), engine.config.batch_size):
                batch = chunks[i : i + engine.config.batch_size]
                if len(batch) == 1:
                    audio, _ = engine.clone_voice(
                        text=batch[0], ref_audio=ref_audio_path, ref_text=character.audition_line
                    )
                    segment_audio.append(audio)
                else:
                    audios, _ = engine.clone_voice(
                        text=batch, ref_audio=ref_audio_path, ref_text=character.audition_line
                    )
                    segment_audio.extend(audios)

            combined = concatenate_audio(segment_audio, SAMPLE_RATE, PARAGRAPH_PAUSE_MS)
            save_chunk_audio(chunk_dir, seg_idx, combined, SAMPLE_RATE)
        except Exception as e:
            print(f"failed segment {seg_idx}: {e}")
            save_chunk_audio(chunk_dir, seg_idx, np.zeros(1, dtype=np.float32), SAMPLE_RATE)

        save_chunk_progress(chunk_dir, seg_idx + 1, total_segments)

    finalize_chunks(chunk_dir, wav_path, total_segments)
    print(f"  -> {wav_path.name}")


def synthesize_dramatized_chapter(
    engine: TTSEngine,
    segments: List[ScriptSegment],
    voices_dir: Path,
    cast_map: dict[str, Character],
) -> tuple[np.ndarray, int]:
    """synthesize a single chapter script using cloned voices with batching."""
    audio_segments = []
    sample_rate = SAMPLE_RATE

    for segment in tqdm(segments, desc="synthesizing segments", leave=False):
        character = cast_map.get(segment.speaker)
        if not character:
            character = cast_map.get("Narrator")

        if not character:
            print(f"warning: unknown speaker {segment.speaker} and no Narrator found")
            continue

        ref_audio_path = voices_dir / f"{character.name}{WAV_EXT}"
        if not ref_audio_path.exists():
            print(f"warning: missing voice for {character.name}")
            continue

        chunks = [c for c in chunk_text(segment.text, engine.config.chunk_size) if c.strip()]
        if not chunks:
            continue

        # batch all chunks for this segment (same voice reference)
        try:
            for i in range(0, len(chunks), engine.config.batch_size):
                batch = chunks[i : i + engine.config.batch_size]
                if len(batch) == 1:
                    audio, sample_rate = engine.clone_voice(
                        text=batch[0], ref_audio=ref_audio_path, ref_text=character.audition_line
                    )
                    audio_segments.append(audio)
                else:
                    audios, sample_rate = engine.clone_voice(
                        text=batch, ref_audio=ref_audio_path, ref_text=character.audition_line
                    )
                    audio_segments.extend(audios)
        except Exception as e:
            print(f"failed to synthesize segment: {e}")

    return concatenate_audio(audio_segments, sample_rate, PARAGRAPH_PAUSE_MS), sample_rate


# CLI Command Wrappers


def cmd_cast(args):
    chapters = None
    if args.chapters:
        from .utils import parse_chapter_range

        chapters = parse_chapter_range(args.chapters)

    run_cast_generation(
        Path(args.workdir),
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
        verbose=args.verbose,
    )


def cmd_audition(args):
    min_appearances = getattr(args, "min_appearances", 0)
    run_auditions(Path(args.workdir), min_appearances=min_appearances, verbose=args.verbose)


def cmd_script(args):
    chapters = None
    if args.chapters:
        from .utils import parse_chapter_range

        chapters = parse_chapter_range(args.chapters)

    run_script_generation(
        Path(args.workdir),
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        chapters=chapters,
        verbose=args.verbose,
    )


def cmd_perform(args):
    chapters = None
    if args.chapters:
        from .utils import parse_chapter_range

        chapters = parse_chapter_range(args.chapters)

    config = TTSConfig(
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        compile_model=not args.no_compile,
        warmup=not args.no_warmup,
        do_sample=not args.greedy,
        temperature=args.temperature,
    )

    min_appearances = getattr(args, "min_appearances", 0)
    run_performance(
        Path(args.workdir),
        chapters,
        config,
        args.pooled,
        min_appearances,
        verbose=args.verbose,
    )


def _normalize_text(text: str) -> str:
    """normalize text for comparison by collapsing whitespace."""
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_boundary_quotes(text: str) -> str:
    """strip quotes and whitespace from text boundaries for comparison."""
    return text.strip(' \t\n"\'""''')


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


def _find_text_in_source(needle: str, haystack: str, start_pos: int = 0) -> tuple[int, int] | None:
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

    matcher = difflib.SequenceMatcher(None, needle_words, haystack_words, autojunk=False)
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


def validate_script(txt_path: Path) -> ValidationResult:
    """validate that script segments match the source text using fuzzy diffing.

    uses difflib to align normalized words between source and script.
    identifies missing text (source words not in script) and hallucinated
    segments (segments with low word match ratio).
    """
    segments, completed, total = load_script(txt_path)
    if not segments:
        return ValidationResult(missing=[(f"no script found for {txt_path.name}", 0)], hallucinated=[])

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

        for current_start, current_end, current_ins, current_offset in missing_ranges[1:]:
            last_start, last_end, last_ins, last_offset = merged[-1]

            # check gap between last_end and current_start
            gap_text = original_text[last_end:current_start]
            
            # merge if:
            # 1. Same insertion index
            # 2. Same split offset (or very close/sequential?) 
            #    Actually, if we merge, we keep the FIRST insertion point (last_ins, last_offset).
            #    But we should only merge if the gap is just punctuation/whitespace.
            #    And typically they will be at the same insertion point if they are contiguous in source
            #    but skipped in script.
            if current_ins == last_ins and current_offset == last_offset and not re.search(r"\w", gap_text):
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

    return ValidationResult(missing=missing_fragments, hallucinated=hallucinated_indices)


def run_validation(
    workdir: Path,
    chapters: list[int] | None = None,
    check_missing: bool = True,
    check_hallucinated: bool = True,
) -> dict[str, ValidationResult]:
    """validate scripts against source text for all chapters.

    returns a dict mapping chapter names to ValidationResult.
    """
    txt_files = sorted(workdir.glob(f"*{TXT_EXT}"))
    if not txt_files:
        print("no text files found in workdir!")
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
        script_path = txt_path.with_suffix(SCRIPT_EXT)
        if not script_path.exists():
            continue
        chapters_to_check.append(txt_path)

    if not chapters_to_check:
        print("no chapters with scripts to validate")
        return {}

    results = {}
    total_missing = 0
    total_hallucinated = 0

    for txt_path in tqdm(chapters_to_check, desc="validating", unit="chapter"):
        result = validate_script(txt_path)
        results[txt_path.name] = result

        if check_missing:
            total_missing += len(result.missing)
        if check_hallucinated:
            total_hallucinated += len(result.hallucinated)

    # print results
    print()
    for txt_path in chapters_to_check:
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
                segments, _, _ = load_script(txt_path)
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
        print(f"\nvalidation passed: all {len(results)} chapters OK")
    else:
        print(f"\nvalidation found {', '.join(summary_parts)} across {len(results)} chapters")

    return results


def cmd_validate(args):
    chapters = None
    if args.chapters:
        from .utils import parse_chapter_range
        chapters = parse_chapter_range(args.chapters)

    # default to checking both if neither flag specified
    check_missing = args.missing or (not args.missing and not args.hallucinated)
    check_hallucinated = args.hallucinated or (not args.missing and not args.hallucinated)

    run_validation(Path(args.workdir), chapters, check_missing, check_hallucinated)




DEFAULT_CONTEXT_CHARS = 500  # default characters of context before/after missing text
DEFAULT_CONTEXT_PARAGRAPHS = 2  # default paragraphs of context before/after missing text


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


def _extract_context_paragraphs(original_text: str, fragment: str, num_paragraphs: int) -> tuple[str, str]:
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
    cast, _ = load_cast(workdir)
    if not cast:
        if (workdir / CAST_FILE).exists():
            print(f"cast file found at {workdir / CAST_FILE} but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return

    txt_files = sorted(workdir.glob(f"*{TXT_EXT}"))
    if not txt_files:
        print("no text files found in workdir!")
        return

    # filter to relevant chapters
    chapters_to_fix = []
    for txt_path in txt_files:
        try:
            chapter_num = int(txt_path.stem.split("_")[0])
        except ValueError:
            continue
        if chapters and chapter_num not in chapters:
            continue
        script_path = txt_path.with_suffix(SCRIPT_EXT)
        if not script_path.exists():
            continue
        chapters_to_fix.append(txt_path)

    if not chapters_to_fix:
        print("no chapters with scripts to fix")
        return

    total_added = 0
    total_removed = 0

    for txt_path in tqdm(chapters_to_fix, desc="fixing", unit="chapter"):
        result = validate_script(txt_path)

        if not result.missing and not result.hallucinated:
            continue

        segments, completed, total = load_script(txt_path)
        original_text = txt_path.read_text(encoding="utf-8")

        # remove hallucinated segments first (in reverse order to preserve indices)
        if fix_hallucinated and result.hallucinated:
            print(f"\n{txt_path.name}: removing {len(result.hallucinated)} hallucinated segment(s)...")
            for idx in sorted(result.hallucinated, reverse=True):
                seg = segments[idx]
                print(f"  removing [{idx}] {seg.speaker}: {seg.text}")
                del segments[idx]
                total_removed += 1
            # checkpoint after removing hallucinations
            save_script(txt_path, segments, total, total)

        # re-validate to get updated missing list after hallucination removal
        if fix_missing:
            result = validate_script(txt_path)
            if result.missing:
                print(f"\n{txt_path.name}: filling {len(result.missing)} missing fragment(s)...")
                # reload segments in case they changed
                segments, completed, total = load_script(txt_path)

                # Sort missing fragments by insertion index and offset in descending order
                # so that insertions/splits don't affect indices of subsequent operations
                missing_sorted = sorted(result.missing, key=lambda x: (x[1], x[2]), reverse=True)

                for fragment, insertion_idx, split_offset in tqdm(missing_sorted, desc="  filling", unit="fragment", leave=False):
                    if verbose:
                        print(f"\n    missing fragment (@ {insertion_idx}+{split_offset}): {fragment}")

                    # Determine insertion point and handle splitting
                    target_idx = insertion_idx
                    if split_offset > 0 and insertion_idx < len(segments):
                        seg = segments[insertion_idx]
                        # only split if offset is within bounds and actually splits text
                        if split_offset < len(seg.text):
                            from copy import deepcopy
                            left_seg = deepcopy(seg)
                            left_seg.text = seg.text[:split_offset].rstrip()
                            
                            right_seg = deepcopy(seg)
                            right_seg.text = seg.text[split_offset:].lstrip()
                            
                            # update segment at insertion_idx with left part
                            segments[insertion_idx] = left_seg
                            # insert right part after
                            segments.insert(insertion_idx + 1, right_seg)
                            
                            # we want to insert the NEW text between left and right
                            target_idx = insertion_idx + 1

                    context_before, context_after = _extract_context(
                        original_text, fragment, context_chars, context_paragraphs
                    )

                    try:
                        new_segments = fix_missing_segment(
                            fragment,
                            context_before,
                            context_after,
                            cast,
                            api_base,
                            api_key,
                            model or "gpt-4o",
                        )

                        if new_segments:
                            if verbose:
                                print("    LLM returned:")
                                for s in new_segments:
                                    print(f"      {s.speaker}: {s.text}")

                            for j, seg in enumerate(new_segments):
                                segments.insert(target_idx + j, seg)
                            
                            # merge newly inserted segments with neighbors if speakers match
                            # 1. merge last new segment with following existing segment
                            if _attempt_merge(segments, target_idx + len(new_segments) - 1):
                                pass # merged with next
                            
                            # 2. merge internal new segments (if LLM returned multiple)
                            # iterate backwards to keep indices valid
                            for j in range(len(new_segments) - 2, -1, -1):
                                _attempt_merge(segments, target_idx + j)

                            # 3. merge preceding existing segment with first new segment
                            if target_idx > 0:
                                _attempt_merge(segments, target_idx - 1)

                            total_added += len(new_segments)
                            # checkpoint after each fragment
                            save_script(txt_path, segments, total, total)

                    except Exception as e:
                        print(f"    failed: {e}")

    # summary
    summary_parts = []
    if fix_missing and total_added > 0:
        summary_parts.append(f"added {total_added} segment(s)")
    if fix_hallucinated and total_removed > 0:
        summary_parts.append(f"removed {total_removed} segment(s)")

    if summary_parts:
        print(f"\nfix complete: {', '.join(summary_parts)}")
    else:
        print("\nno fixes needed")


def cmd_fix(args):
    chapters = None
    if args.chapters:
        from .utils import parse_chapter_range
        chapters = parse_chapter_range(args.chapters)

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
    min_appearances: int = 0,
    verbose: bool = False,
) -> None:
    """run full dramatization pipeline."""
    cast = run_cast_generation(workdir, api_base, api_key, model, chapters, verbose=verbose)
    # generate scripts first to count appearances before auditions
    run_script_generation(workdir, api_base, api_key, model, chapters, verbose=verbose)
    # now run auditions with appearance counts available
    run_auditions(workdir, min_appearances=min_appearances, verbose=verbose)
    run_performance(workdir, chapters, tts_config, pooled, min_appearances, verbose=verbose)
