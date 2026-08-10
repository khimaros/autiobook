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
from .config import (
    CAST_FILE,
    SAMPLE_RATE,
    VOICE_DESIGN_MODEL,
    WAV_EXT,
)
from .dramatize import load_cast
from .llm import Character
from .resume import ResumeManager, compute_hash, get_command_dir
from .utils import create_tts_engine, prompt_choice

AUDITION_COMMAND = "audition"
# resume key for the roster a directed session was approved against. leading
# underscores keep it clear of character names, which share this namespace.
CAST_APPROVAL_KEY = "__cast__"


def audition_task_hash(name: str, instruct: str, text: str) -> str:
    """resume hash for one character's base voice take."""
    return compute_hash({"name": name, "description": instruct, "text": text})


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


def recorded_audition_lines(workdir: Path) -> dict[str, str]:
    """per character, the line their audition wav was actually rendered with.

    the cast's audition_line can be reworded by a later merge, and `--accept`
    marks an existing wav fresh without re-rendering it, so the cast is not a
    safe source for ref_text. the resume entry is: it is written at the moment
    the audio is produced.
    """
    resume = ResumeManager.for_command(workdir, AUDITION_COMMAND)
    return {
        name: str(entry["audition_line"])
        for name, entry in resume.state.items()
        if isinstance(entry, dict) and entry.get("audition_line")
    }


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
    accept: bool = False,
) -> None:
    """generate the per-character base voice (description only).

    modes:
      preset_voices=True, directed=False: auto round-robin assign backend voices
      preset_voices=True, directed=True: interactive casting loop with backend voices
      accept=True: skip regeneration; mark existing wavs as fresh under current
        cast hashes (useful after tweaking descriptions without wanting to redo
        the audio)
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

    if accept:
        _accept_existing(workdir, cast, voices_dir, resume, preset_voices=preset_voices)
        return

    if directed:
        pending = _pending_characters(
            workdir, cast, voices_dir, preset_voices, force, audition_line
        )
        if not _confirm_cast(cast, pending, resume):
            print("audition: cancelled.")
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
        text = char.audition_text(audition_line)
        instruct = char.voice_prompt()

        task_hash = audition_task_hash(char.name, instruct, text)

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


def _pending_characters(
    workdir: Path,
    cast: List[Character],
    voices_dir: Path,
    preset_voices: bool,
    force: bool,
    audition_line: str | None,
) -> List[Character]:
    """characters a directed session would actually stop on.

    mirrors the skip each directed path applies per character, so the roster
    covers the work ahead rather than voices that are already approved.
    """
    if force:
        return list(cast)
    if preset_voices:
        from .casting import load_voices

        voices = load_voices(workdir)
        return [c for c in cast if not voices.get(c.name)]
    resume = ResumeManager.for_command(workdir, AUDITION_COMMAND)
    pending = []
    for char in cast:
        text = char.audition_text(audition_line)
        task_hash = audition_task_hash(char.name, char.voice_prompt(), text)
        if (voices_dir / f"{char.name}{WAV_EXT}").exists() and resume.is_fresh(
            char.name, task_hash
        ):
            continue
        pending.append(char)
    return pending


def _confirm_cast(
    cast: List[Character], pending: List[Character], resume: ResumeManager
) -> bool:
    """show the pending roster and confirm before a directed session starts.

    a directed run walks every character in turn and can take an hour, so a
    cast with junk entries or duplicates is worth catching before the first
    take rather than at character 30. approval is recorded against the cast
    itself, so resuming an interrupted session goes straight back to work --
    only an edited cast is worth a second look.
    """
    if not pending:
        return True
    roster_hash = compute_hash(
        [
            {
                "name": c.name,
                "description": c.description,
                "voice": c.voice,
                "aliases": c.aliases or [],
            }
            for c in cast
        ]
    )
    if resume.is_fresh(CAST_APPROVAL_KEY, roster_hash):
        return True

    approved = len(cast) - len(pending)
    suffix = f" ({approved} already approved)" if approved else ""
    print(f"\ncast: {len(pending)} characters to audition{suffix}")
    for i, char in enumerate(pending, 1):
        aliases = f" (also: {', '.join(char.aliases)})" if char.aliases else ""
        print(f"  {i:3d}. {char.name}{aliases}")
        print(f"       {char.description}")
    ans = prompt_choice(
        f"\nproceed with these {len(pending)} characters? [y]es / [q]uit: "
    )
    if ans not in ("y", "yes", ""):
        return False
    resume.update(CAST_APPROVAL_KEY, roster_hash)
    resume.save()
    return True


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
    and prompt y/n/e/s/q. 'next' picks a new random seed; 'edit' revises
    the description and retries.
    """
    import random
    from threading import Event, Lock, Thread

    from .casting import _play_pcm_stream, _play_wav_async, _stop_playback
    from .dramatize import save_cast

    resume = ResumeManager.for_command(workdir, AUDITION_COMMAND, force=force)
    takes_dir = voices_dir / "takes"
    takes_dir.mkdir(parents=True, exist_ok=True)

    # [n]ext and [e]dit walk config.seed to fresh random values, and the pregen
    # worker sets it as a side effect. both outlive the character they were
    # rolled for, so each character starts from the configured seed again --
    # otherwise a character's first take depends on how many times the previous
    # character was skipped past.
    base_seed = int(getattr(config, "seed", 0) or 0)

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
            # one-shot seed for the next take, so a character's first take can
            # start from the configured seed the way the streaming path does.
            self.next_seed: int | None = None
            self._lock = Lock()

        def set_target(
            self, text_: str, instruct_: str, first_seed: int | None = None
        ) -> None:
            """declare the desired (text, instruct). flushes buffer on instruct change.

            first_seed applies to the take generated after the flush; later
            takes roll their own, matching [n]ext on the streaming path."""
            with self._lock:
                if instruct_ != self.instruct:
                    self.ready = []
                    self.next_seed = first_seed
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
                # `or` covers an unset configured seed (0), which the streaming
                # path also answers with a random one
                seed = self.next_seed or random.randint(1, 2**31 - 1)
                self.next_seed = None

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

        def try_take(self, instruct_: str) -> tuple[int, Any, int] | None:
            """non-blocking pop: return a ready take or None immediately.

            used when a streaming foreground path is available — we prefer
            fast TTFA over joining the pregen worker's non-streaming synth.
            """
            self.set_target(self.text or "", instruct_)
            with self._lock:
                if self.instruct != instruct_ or not self.ready:
                    item = None
                else:
                    item = self.ready.pop(0)
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

    # streaming path bypasses pregen: pregen's non-streaming worker holds
    # engine_lock, which would block streaming requests for the full duration
    # of a pregen synth. streaming TTFA (~1 batch) is fast enough that we
    # don't need a zero-wait buffer for successive takes.
    stream_fn = getattr(engine, "design_voice_stream", None)
    streaming_available = callable(stream_fn) and getattr(engine, "streaming", False)

    accepted = 0
    skipped = 0
    # bound before the loop: a resumed run skips every character, so nothing
    # inside the body executes and the post-loop check still has to read it.
    quit_requested = False
    for char in cast:
        config.seed = base_seed
        final = voices_dir / f"{char.name}{WAV_EXT}"
        text = char.audition_text(audition_line)
        instruct = char.voice_prompt()

        task_hash = audition_task_hash(char.name, instruct, text)
        if not force and final.exists() and resume.is_fresh(char.name, task_hash):
            skipped += 1
            continue

        print(f"\n=== {char.name} ===")
        print(f"  description: {instruct}")
        print(f"  line: {text!r}")

        # takes: each entry [seed, instruct, audio|None, sr|None, path|None].
        # uncached entries (audio is None) get re-streamed on revisit using
        # the recorded seed — so 'p' returns to a phantom take that the user
        # navigated away from before its synth completed.
        takes: list[list[Any]] = []
        cursor = -1
        # what to do at top of next iteration: "new" (generate at end),
        # "restream" (re-stream takes[cursor]), or None (just prompt).
        next_action: str | None = "new"
        pregen: PregenQueue | None = None
        if not streaming_available:
            pregen = PregenQueue(max_depth=PREGEN_DEPTH)
            pregen.set_target(text, instruct, first_seed=base_seed)

        # pending streaming synth: (thread, take_index, result, live_proc, cancel)
        pending: tuple[Thread, int, dict, Any, Event] | None = None

        def _start_stream(seed: int, synth_instruct: str, take_index: int) -> None:
            """spawn a background streaming synth that pipes pcm to ffplay."""
            nonlocal pending, playback
            stop()
            live_proc = _play_pcm_stream(SAMPLE_RATE)
            use_live = live_proc is not None and live_proc.stdin is not None
            if use_live:
                playback = live_proc
            result: dict[str, Any] = {}
            cancel_event = Event()

            def _synth(
                _live_proc=live_proc,
                _use_live=use_live,
                _seed=seed,
                _instruct=synth_instruct,
                _result=result,
                _cancel=cancel_event,
            ) -> None:
                try:
                    with engine_lock:
                        config.seed = _seed
                        if _use_live:

                            def _on_chunk(b: bytes) -> None:
                                try:
                                    _live_proc.stdin.write(b)
                                    _live_proc.stdin.flush()
                                except (BrokenPipeError, OSError):
                                    pass

                            assert stream_fn is not None
                            a, s = stream_fn(
                                text=text,
                                instruct=_instruct,
                                on_chunk=_on_chunk,
                                cancel=_cancel,
                            )
                        else:
                            a, s = engine.design_voice(text=text, instruct=_instruct)
                    _result["audio"] = a
                    _result["sr"] = s
                except Exception as e:
                    _result["error"] = e
                finally:
                    if (
                        _use_live
                        and _live_proc is not None
                        and _live_proc.stdin is not None
                    ):
                        try:
                            _live_proc.stdin.close()
                        except OSError:
                            pass

            t = Thread(target=_synth, daemon=True)
            t.start()
            pending = (t, take_index, result, live_proc, cancel_event)

        while True:
            if next_action == "new":
                got = pregen.take(instruct) if pregen is not None else None
                if got is not None:
                    seed, audio, sr = got
                    if verbose:
                        print(f"  seed={seed}")
                    take_path = takes_dir / f"{char.name}__{seed}{WAV_EXT}"
                    sf.write(str(take_path), audio, sr)
                    takes.append([seed, instruct, audio, sr, take_path])
                    cursor = len(takes) - 1
                    if pregen is not None:
                        pregen.ensure_running()
                    play(take_path)
                else:
                    seed = int(getattr(config, "seed", 0) or 0) or random.randint(
                        1, 2**31 - 1
                    )
                    if verbose:
                        print(f"  seed={seed}")
                    takes.append([seed, instruct, None, None, None])
                    cursor = len(takes) - 1
                    _start_stream(seed, instruct, cursor)
            elif next_action == "restream":
                t_seed, t_instruct = takes[cursor][0], takes[cursor][1]
                if verbose:
                    print(f"  seed={t_seed} (restream)")
                _start_stream(t_seed, t_instruct, cursor)
            next_action = None

            pos = f"{cursor + 1}/{len(takes)}"
            ans = prompt_choice(
                f"  [{pos}] [y]es / [n]ext / [p]rev / [r]eplay / [e]dit / [s]kip / [q]uit: "
            )
            # resolve any pending streaming synth. y/r need full audio so we
            # wait. nav actions (n/p/d) don't wait — but if the synth already
            # finished, cache it. s/q cancel and drop. cached results are
            # written into takes[take_index] so subsequent revisits replay
            # from disk instead of re-streaming.
            if pending is not None:
                _t, _idx, _result, _live_proc, _cancel = pending
                wait = ans in ("y", "yes", "r", "replay")
                drop = ans in ("s", "skip", "q", "quit")
                if drop or not wait:
                    _cancel.set()
                if wait and _t.is_alive():
                    print("  (waiting for synthesis to finish...)")
                    _t.join()
                done = not _t.is_alive()
                cache = (
                    not drop and done and "error" not in _result and "audio" in _result
                )
                pending = None
                if cache:
                    seed_c = takes[_idx][0]
                    audio = _result["audio"]
                    sr = _result["sr"]
                    take_path = takes_dir / f"{char.name}__{seed_c}{WAV_EXT}"
                    sf.write(str(take_path), audio, sr)
                    takes[_idx][2] = audio
                    takes[_idx][3] = sr
                    takes[_idx][4] = take_path
                else:
                    if _live_proc is not None:
                        _stop_playback(_live_proc)
                    if wait and done and "error" in _result:
                        print(f"  failed: {_result['error']}")
            stop()
            cur_seed, cur_instruct, cur_audio, cur_sr, cur_path = takes[cursor]
            if ans in ("y", "yes"):
                if cur_audio is None:
                    print("  (synth incomplete; re-running to accept...)")
                    _start_stream(cur_seed, cur_instruct, cursor)
                    assert pending is not None
                    _t, _idx, _result, _live_proc, _cancel = pending
                    _t.join()
                    pending = None
                    stop()
                    if "audio" not in _result:
                        err = _result.get("error", "unknown")
                        print(f"  failed: {err}")
                        continue
                    cur_audio = _result["audio"]
                    cur_sr = _result["sr"]
                    cur_path = takes_dir / f"{char.name}__{cur_seed}{WAV_EXT}"
                    sf.write(str(cur_path), cur_audio, cur_sr)
                    takes[cursor][2] = cur_audio
                    takes[cursor][3] = cur_sr
                    takes[cursor][4] = cur_path
                accept_hash = audition_task_hash(char.name, cur_instruct, text)
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
                if cur_instruct != char.voice_prompt():
                    char.voice = cur_instruct
                    save_cast(workdir, cast)
                    print("  updated cast voice")
                accepted += 1
                print(f"  accepted (seed={cur_seed})")
                break
            if ans in ("r", "replay"):
                if cur_audio is not None and cur_path is not None:
                    play(cur_path)
                else:
                    next_action = "restream"
                if pregen is not None:
                    pregen.ensure_running()
                continue
            if ans in ("p", "prev"):
                if cursor > 0:
                    cursor -= 1
                    if takes[cursor][2] is not None:
                        play(takes[cursor][4])
                    else:
                        next_action = "restream"
                else:
                    print("  (no earlier take)")
                if pregen is not None:
                    pregen.ensure_running()
                continue
            if ans in ("e", "edit"):
                new_desc = _edit_description(instruct)
                if new_desc != instruct:
                    instruct = new_desc
                    print(f"  voice: {instruct}")
                    if pregen is not None:
                        pregen.set_target(text, instruct)
                config.seed = random.randint(1, 2**31 - 1)
                next_action = "new"
                continue
            if ans in ("s", "skip"):
                break
            if ans in ("q", "quit"):
                quit_requested = True
                break
            # [n]ext or empty: forward through history, or generate new at end
            if cursor < len(takes) - 1:
                cursor += 1
                if takes[cursor][2] is not None:
                    play(takes[cursor][4])
                else:
                    next_action = "restream"
                if pregen is not None:
                    pregen.ensure_running()
            else:
                config.seed = random.randint(1, 2**31 - 1)
                next_action = "new"

        stop()
        # let any background pregen finish before moving to next character
        if pregen is not None:
            pregen.join()
        if quit_requested:
            break

    resume.save()
    print(f"audition: {accepted} accepted, {skipped} skipped")
    if quit_requested:
        # halt the pipeline rather than silently advancing to script.
        raise KeyboardInterrupt


def _accept_existing(
    workdir: Path,
    cast: List[Character],
    voices_dir: Path,
    resume: ResumeManager,
    preset_voices: bool = False,
) -> None:
    """mark existing audition wavs as fresh under current cast hashes."""
    voices_map: dict[str, str] = {}
    if preset_voices:
        from .casting import load_voices

        voices_map = load_voices(workdir)

    updated = 0
    missing: list[str] = []
    for char in cast:
        wav_path = voices_dir / f"{char.name}{WAV_EXT}"
        if not wav_path.exists():
            missing.append(char.name)
            continue

        prior = (
            resume.state.get(char.name, {}) if isinstance(resume.state, dict) else {}
        )
        prior_entry = prior if isinstance(prior, dict) else {}

        if preset_voices:
            voice_id = voices_map.get(char.name) or prior_entry.get("voice_id")
            if not voice_id:
                missing.append(f"{char.name} (no voice_id)")
                continue
            task_hash = compute_hash(
                {
                    "name": char.name,
                    "voice_id": voice_id,
                    "text": char.audition_text(),
                    "mode": "preset",
                }
            )
            resume.update(
                char.name,
                task_hash,
                character=char.name,
                voice_id=voice_id,
                audition_line=char.audition_text(),
                wav_sha256=wav_sha256(wav_path),
            )
        else:
            task_hash = audition_task_hash(
                char.name, char.voice_prompt(), char.audition_text()
            )
            resume.update(
                char.name,
                task_hash,
                character=char.name,
                prompt=char.voice_prompt(),
                audition_line=char.audition_text(),
                seed=int(prior_entry.get("seed", 0) or 0),
                wav_sha256=wav_sha256(wav_path),
            )
        updated += 1

    resume.save()
    print(f"audition: accepted {updated} existing voices")
    if missing:
        print(
            f"audition: {len(missing)} missing — re-run without --accept "
            f"to fill in: {', '.join(missing)}"
        )


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
                "text": char.audition_text(),
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
                audio, sr = engine.synthesize(char.audition_text())
            finally:
                engine.config.speaker = original
            sf_mod.write(str(final), audio, sr)

        resume.update(
            char.name,
            task_hash,
            character=char.name,
            voice_id=voice_id,
            audition_line=char.audition_text(),
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
        from .utils import _build_http_config, _resolve_tts_model, tts_endpoint

        if not tts_endpoint(args)[0]:
            raise RuntimeError(
                "--preset-voices requires --api-base or --tts-api-base "
                "(or set OPENAI_BASE_URL)"
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
        accept=getattr(args, "accept", False),
    )
