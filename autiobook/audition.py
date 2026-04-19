"""audition phase: generate the canonical base voice for each cast member.

uses design_voice with the character description only (no emotion hints) and
saves to audition/{name}.wav. this base file is the per-character voice
identity, used as a fallback ref clip during perform when an emotion variant
is missing. honors --callback and archives rejects under audition/rejected/.
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

AUDITION_COMMAND = "audition"


def _edit_description(initial: str) -> str:
    """open $EDITOR (or $VISUAL, falling back to nano/vi) with initial text.

    returns the edited text stripped of trailing whitespace, or initial if
    the editor is unavailable or the user blanks the buffer.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        for cmd in ("nano", "vim", "vi"):
            if shutil.which(cmd):
                editor = cmd
                break
    if not editor:
        try:
            ans = input(f"  new description [{initial}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return initial
        return ans or initial

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(initial)
        tmp_path = tf.name
    try:
        subprocess.run([editor, tmp_path], check=False)
        with open(tmp_path, encoding="utf-8") as f:
            edited = f.read().strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return edited or initial


def recorded_seed(workdir: Path, character_name: str) -> int:
    """return the seed recorded by audition for a character, or 0 if none."""
    resume = ResumeManager.for_command(workdir, AUDITION_COMMAND)
    return _seed_from_entry(resume.state.get(character_name))


def _seed_from_entry(entry: Any) -> int:
    if isinstance(entry, dict):
        try:
            return int(entry.get("seed", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def run_audition(
    workdir: Path,
    cast: List[Character] | None = None,
    verbose: bool = False,
    force: bool = False,
    audition_line: str | None = None,
    config: Any = None,
    callback: bool = False,
    preset_voices: bool = False,
    directed: bool = False,
) -> None:
    """generate the per-character base voice (description only).

    modes:
      preset_voices=True, directed=False: auto round-robin assign backend voices
      preset_voices=True, directed=True: interactive casting loop with backend voices
      otherwise: generate voices via voice-design (default)
    """
    if cast is None:
        cast = load_cast(workdir)

    voices_dir = get_command_dir(workdir, AUDITION_COMMAND)
    resume = ResumeManager.for_command(workdir, AUDITION_COMMAND, force=force)

    if not cast:
        cast_path = get_command_dir(workdir, "cast") / CAST_FILE
        if cast_path.exists():
            print(f"cast file found at {cast_path} but contains no characters.")
        else:
            print("no cast found. run 'cast' command first.")
        return

    if preset_voices:
        _run_preset(workdir, cast, config, voices_dir, force=force, directed=directed)
        return

    if config is None:
        from .tts import TTSConfig

        config = TTSConfig(model_name=VOICE_DESIGN_MODEL)
    engine = create_tts_engine(config)

    if directed:
        _run_directed_design(
            workdir,
            cast,
            engine,
            config,
            voices_dir,
            force=force,
            audition_line=audition_line,
            verbose=verbose,
        )
        return

    print(f"auditioning {len(cast)} characters...")

    generated_count = 0
    skipped_count = 0

    for char in tqdm(cast, desc="auditioning voices"):
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
                    reject_dir=get_reject_dir(workdir, AUDITION_COMMAND),
                    metadata={
                        "phase": "audition",
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
            print(f"failed to audition {char.name}: {e}")
            raise

    resume.save()
    if generated_count == 0 and skipped_count > 0:
        print(f"audition: all {skipped_count} voices up to date.")
    else:
        print(f"audition: {generated_count} generated, {skipped_count} skipped")


def _run_directed_design(
    workdir: Path,
    cast: List[Character],
    engine: Any,
    config: Any,
    voices_dir: Path,
    force: bool = False,
    audition_line: str | None = None,
    verbose: bool = False,
) -> None:
    """interactive voice-design casting: regenerate takes until user approves.

    for each character, synthesize a take with (description, seed), play it,
    and prompt y/n/d/s/q. 'next' picks a new random seed; 'describe' edits
    the description and retries.
    """
    import random
    from threading import Lock, Thread

    from .casting import _play_wav_async, _prompt, _stop_playback
    from .dramatize import save_cast

    resume = ResumeManager.for_command(workdir, AUDITION_COMMAND, force=force)
    takes_dir = voices_dir / "takes"
    takes_dir.mkdir(parents=True, exist_ok=True)

    engine_lock = Lock()

    PREGEN_DEPTH = 3

    class PregenQueue:
        """buffered background pregen: keeps up to PREGEN_DEPTH ready takes.

        single worker at a time (serialized via engine_lock); chains itself
        until the buffer is full. on instruct change the buffer is flushed and
        an in-flight take's result is discarded.
        """

        def __init__(self, max_depth: int) -> None:
            self.max_depth = max_depth
            self.text: str = ""
            self.instruct: str = ""
            self.ready: list[tuple[int, Any, int]] = []
            self.thread: Thread | None = None
            self._lock = Lock()

        def set_target(self, text_: str, instruct_: str) -> None:
            """declare the desired (text, instruct). flushes buffer on instruct change."""
            with self._lock:
                if instruct_ != self.instruct:
                    self.ready = []
                self.text = text_
                self.instruct = instruct_
            self.ensure_running()

        def ensure_running(self) -> None:
            """spawn a worker if none is in flight and buffer has headroom."""
            with self._lock:
                if self.thread is not None and self.thread.is_alive():
                    return
                if not self.instruct or len(self.ready) >= self.max_depth:
                    return
                target_text = self.text
                target_instruct = self.instruct
                seed = random.randint(1, 2**31 - 1)

            def run() -> None:
                audio: Any = None
                sr = 0
                try:
                    with engine_lock:
                        config.seed = seed
                        audio, sr = engine.design_voice(
                            text=target_text, instruct=target_instruct
                        )
                except Exception:
                    audio = None
                with self._lock:
                    if (
                        audio is not None
                        and self.instruct == target_instruct
                        and len(self.ready) < self.max_depth
                    ):
                        self.ready.append((seed, audio, sr))
                self.ensure_running()

            self.thread = Thread(target=run, daemon=True)
            self.thread.start()

        def take(self, instruct_: str) -> tuple[int, Any, int] | None:
            """pop a ready take for instruct_, waiting on the in-flight worker.

            returns None if buffer is empty AND no worker is running for this
            instruct (caller should foreground-synth). always (re)kicks the
            worker before returning.
            """
            self.set_target(self.text or "", instruct_)
            while True:
                with self._lock:
                    if self.instruct != instruct_:
                        return None
                    if self.ready:
                        item = self.ready.pop(0)
                        break
                    thread = self.thread
                    if thread is None or not thread.is_alive():
                        return None
                thread.join()
            self.ensure_running()
            return item

        def join(self) -> None:
            if self.thread:
                self.thread.join()
                self.thread = None

    playback: Any = None

    def play(path: Path) -> None:
        nonlocal playback
        _stop_playback(playback)
        playback = _play_wav_async(path)

    def stop() -> None:
        nonlocal playback
        _stop_playback(playback)
        playback = None

    accepted = 0
    skipped = 0
    for char in cast:
        final = voices_dir / f"{char.name}{WAV_EXT}"
        text = audition_line or char.audition_line
        instruct = char.description

        task_data = {"name": char.name, "description": instruct, "text": text}
        task_hash = compute_hash(task_data)
        if not force and final.exists() and resume.is_fresh(char.name, task_hash):
            skipped += 1
            continue

        print(f"\n=== {char.name} ===")
        print(f"  description: {instruct}")
        print(f"  line: {text!r}")

        # takes: list of (seed, instruct, audio, sr, path). cursor points at current.
        takes: list[tuple[int, str, Any, int, Path]] = []
        cursor = -1
        quit_requested = False
        generate_next = True
        pregen = PregenQueue(max_depth=PREGEN_DEPTH)
        pregen.set_target(text, instruct)

        while True:
            if generate_next:
                got = pregen.take(instruct)
                if got is not None:
                    seed, audio, sr = got
                    if verbose:
                        print(f"  seed={seed} (pregen)")
                else:
                    seed = int(getattr(config, "seed", 0) or 0) or random.randint(
                        1, 2**31 - 1
                    )
                    if verbose:
                        print(f"  seed={seed}")
                    try:
                        with engine_lock:
                            config.seed = seed
                            audio, sr = engine.design_voice(
                                text=text, instruct=instruct
                            )
                    except Exception as e:
                        print(f"  failed: {e}")
                        ans = _prompt("  [n]ext / [s]kip / [q]uit: ")
                        if ans in ("q", "quit"):
                            quit_requested = True
                            break
                        if ans in ("s", "skip"):
                            break
                        config.seed = random.randint(1, 2**31 - 1)
                        continue
                take_path = takes_dir / f"{char.name}__{seed}{WAV_EXT}"
                sf.write(str(take_path), audio, sr)
                takes.append((seed, instruct, audio, sr, take_path))
                cursor = len(takes) - 1
                generate_next = False
                # keep the pregen buffer filling; play current concurrently
                pregen.ensure_running()
                play(take_path)

            cur_seed, cur_instruct, cur_audio, cur_sr, cur_path = takes[cursor]
            pos = f"{cursor + 1}/{len(takes)}"
            ans = _prompt(
                f"  [{pos}] [y]es / [n]ext / [p]rev / [r]eplay / [d]escribe / [s]kip / [q]uit: "
            )
            stop()
            if ans in ("y", "yes"):
                accept_hash = compute_hash(
                    {"name": char.name, "description": cur_instruct, "text": text}
                )
                sf.write(str(final), cur_audio, cur_sr)
                resume.update(
                    char.name,
                    accept_hash,
                    character=char.name,
                    prompt=cur_instruct,
                    audition_line=text,
                    seed=cur_seed,
                    wav_sha256=wav_sha256(final),
                )
                resume.save()
                if cur_instruct != char.description:
                    char.description = cur_instruct
                    save_cast(workdir, cast)
                    print("  updated cast description")
                accepted += 1
                print(f"  accepted (seed={cur_seed})")
                break
            if ans in ("r", "replay"):
                play(cur_path)
                pregen.ensure_running()
                continue
            if ans in ("p", "prev"):
                if cursor > 0:
                    cursor -= 1
                    play(takes[cursor][4])
                else:
                    print("  (no earlier take)")
                pregen.ensure_running()
                continue
            if ans in ("d", "describe"):
                new_desc = _edit_description(instruct)
                if new_desc != instruct:
                    instruct = new_desc
                    print(f"  description: {instruct}")
                    # flush stale buffer and start refilling for the new instruct
                    pregen.set_target(text, instruct)
                config.seed = random.randint(1, 2**31 - 1)
                generate_next = True
                continue
            if ans in ("s", "skip"):
                break
            if ans in ("q", "quit"):
                quit_requested = True
                break
            # [n]ext or empty: forward through history, or generate new at end
            if cursor < len(takes) - 1:
                cursor += 1
                play(takes[cursor][4])
                pregen.ensure_running()
            else:
                config.seed = random.randint(1, 2**31 - 1)
                generate_next = True

        stop()
        # let any background pregen finish before moving to next character
        pregen.join()
        if quit_requested:
            break

    resume.save()
    print(f"audition: {accepted} accepted, {skipped} skipped")


def _run_preset(
    workdir: Path,
    cast: List[Character],
    config: Any,
    voices_dir: Path,
    force: bool = False,
    directed: bool = False,
) -> None:
    """assign preset backend voices to characters and render audition wavs."""
    from .casting import load_voices, run_casting, save_voices
    from .tts_http import HTTPTTSConfig, HTTPTTSEngine

    if not isinstance(config, HTTPTTSConfig):
        raise RuntimeError(
            "--preset-voices requires an http tts backend (pass --api-base or set OPENAI_BASE_URL)"
        )

    engine = HTTPTTSEngine(config)

    if directed:
        assignments = run_casting(workdir, cast, engine, force=force)
    else:
        assignments = _assign_round_robin(engine, cast, workdir, force=force)
        save_voices(workdir, assignments)

    if not assignments:
        print("audition: no voices assigned")
        return

    resume = ResumeManager.for_command(workdir, AUDITION_COMMAND, force=force)
    for char in cast:
        voice_id = assignments.get(char.name)
        if not voice_id:
            continue
        final = voices_dir / f"{char.name}{WAV_EXT}"
        task_hash = compute_hash(
            {
                "name": char.name,
                "voice_id": voice_id,
                "text": char.audition_line,
                "mode": "preset",
            }
        )
        if not force and final.exists() and resume.is_fresh(char.name, task_hash):
            continue

        # reuse cached preview if present (from directed mode), else synthesize
        from .casting import _preview_path

        preview = _preview_path(workdir, char.name, voice_id)
        if preview.exists():
            final.write_bytes(preview.read_bytes())
        else:
            import soundfile as sf_mod  # type: ignore

            original = engine.config.speaker
            try:
                engine.config.speaker = voice_id
                audio, sr = engine.synthesize(char.audition_line)
            finally:
                engine.config.speaker = original
            sf_mod.write(str(final), audio, sr)

        resume.update(
            char.name,
            task_hash,
            character=char.name,
            voice_id=voice_id,
            audition_line=char.audition_line,
            wav_sha256=wav_sha256(final),
        )
    resume.save()
    print(f"audition: {len(assignments)} characters cast")
    _ = load_voices  # keep symbol imported for clarity


def _assign_round_robin(
    engine: Any, cast: List[Character], workdir: Path, force: bool = False
) -> dict[str, str]:
    """deterministically assign backend voices to characters.

    narrator gets the first voice, others cycle through the rest in cast order.
    existing voices.json entries are preserved unless force=True.
    """
    from .casting import load_voices

    available = engine.list_voices()
    if not available:
        raise RuntimeError("no preset voices returned from backend")

    existing = {} if force else load_voices(workdir)
    assigned: dict[str, str] = dict(existing)

    # narrator takes first voice
    narrator = next((c for c in cast if c.name == "Narrator"), None)
    others = [c for c in cast if c.name != "Narrator"]
    rest_voices = [v for v in available if v != available[0]] or available

    if narrator and narrator.name not in assigned:
        assigned[narrator.name] = available[0]
    for i, char in enumerate(others):
        if char.name in assigned:
            continue
        assigned[char.name] = rest_voices[i % len(rest_voices)]

    print(f"audition: round-robin assigned {len(assigned)} voices")
    return assigned


def cmd_audition(args):
    from .utils import get_design_config

    config = get_design_config(args)
    preset_voices = getattr(args, "preset_voices", False)
    directed = getattr(args, "directed", False)
    if preset_voices:
        # preset mode needs an http backend and the custom-voice tts model
        from .config import DEFAULT_MODEL
        from .utils import _build_http_config, _resolve_tts_model

        if not getattr(args, "api_base", None):
            raise RuntimeError(
                "--preset-voices requires --api-base (or set OPENAI_BASE_URL)"
            )
        model = _resolve_tts_model(args, DEFAULT_MODEL)
        config = _build_http_config(args, model)
    run_audition(
        Path(args.workdir),
        verbose=args.verbose,
        force=args.force,
        audition_line=getattr(args, "audition_line", None),
        config=config,
        callback=getattr(args, "callback", False),
        preset_voices=preset_voices,
        directed=directed,
    )
