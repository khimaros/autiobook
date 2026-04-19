"""introduce phase: generate the canonical base voice for each cast member.

uses design_voice with the character description only (no emotion hints) and
saves to introduce/{name}.wav. this base file is the per-character voice
identity, used as a fallback ref clip during perform when an emotion variant
is missing. honors --callback and archives rejects under introduce/rejected/.
"""

from pathlib import Path
from typing import Any, List

import soundfile as sf  # type: ignore
from tqdm import tqdm  # type: ignore

from .audio import wav_sha256
from .config import CAST_FILE, VOICE_DESIGN_MODEL, WAV_EXT
from .dramatize import load_cast
from .llm import Character
from .resume import ResumeManager, compute_hash, get_command_dir
from .utils import create_tts_engine

INTRODUCE_COMMAND = "introduce"


def recorded_seed(workdir: Path, character_name: str) -> int:
    """return the seed recorded by introduce for a character, or 0 if none."""
    resume = ResumeManager.for_command(workdir, INTRODUCE_COMMAND)
    return _seed_from_entry(resume.state.get(character_name))


def _seed_from_entry(entry: Any) -> int:
    if isinstance(entry, dict):
        try:
            return int(entry.get("seed", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def run_introduce(
    workdir: Path,
    cast: List[Character] | None = None,
    verbose: bool = False,
    force: bool = False,
    audition_line: str | None = None,
    config: Any = None,
    callback: bool = False,
) -> None:
    """generate the per-character base voice (description only)."""
    if cast is None:
        cast = load_cast(workdir)

    voices_dir = get_command_dir(workdir, INTRODUCE_COMMAND)
    resume = ResumeManager.for_command(workdir, INTRODUCE_COMMAND, force=force)

    if not cast:
        cast_path = get_command_dir(workdir, "cast") / CAST_FILE
        if cast_path.exists():
            print(f"cast file found at {cast_path} but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return

    if config is None:
        from .tts import TTSConfig

        config = TTSConfig(model_name=VOICE_DESIGN_MODEL)
    engine = create_tts_engine(config)

    print(f"introducing {len(cast)} characters...")

    generated_count = 0
    skipped_count = 0

    for char in tqdm(cast, desc="introducing voices"):
        wav_path = voices_dir / f"{char.name}{WAV_EXT}"
        text = audition_line or char.audition_line
        instruct = char.description

        task_data = {
            "name": char.name,
            "description": instruct,
            "text": text,
        }
        task_hash = compute_hash(task_data)

        if not force and wav_path.exists() and resume.is_fresh(char.name, task_hash):
            skipped_count += 1
            continue

        if verbose:
            tqdm.write(f"  {char.name}: '{text[:60]}'")

        try:
            if callback:
                from .callback import generate_with_callback
                from .retake import get_reject_dir

                audio, sr = generate_with_callback(
                    lambda: engine.design_voice(text=text, instruct=instruct),
                    engine,
                    label=char.name,
                    verbose=verbose,
                    reject_dir=get_reject_dir(workdir, INTRODUCE_COMMAND),
                    metadata={
                        "phase": "introduce",
                        "character": char.name,
                        "text": text,
                        "instruct": instruct,
                    },
                )
            else:
                audio, sr = engine.design_voice(text=text, instruct=instruct)
            sf.write(str(wav_path), audio, sr)
            resume.update(
                char.name,
                task_hash,
                character=char.name,
                prompt=instruct,
                audition_line=text,
                seed=int(getattr(config, "seed", 0) or 0),
                wav_sha256=wav_sha256(wav_path),
            )
            resume.save()
            generated_count += 1
        except Exception as e:
            print(f"failed to introduce {char.name}: {e}")
            raise

    resume.save()
    if generated_count == 0 and skipped_count > 0:
        print(f"introduce: all {skipped_count} voices up to date.")
    else:
        print(f"introduce: {generated_count} generated, {skipped_count} skipped")


def cmd_introduce(args):
    from .utils import get_design_config

    config = get_design_config(args)
    run_introduce(
        Path(args.workdir),
        verbose=args.verbose,
        force=args.force,
        audition_line=getattr(args, "audition_line", None),
        config=config,
        callback=getattr(args, "callback", False),
    )
