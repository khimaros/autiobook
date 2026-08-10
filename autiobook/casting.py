"""interactive casting: assign preset voices to characters one take at a time."""

import json
import shutil
import subprocess
from pathlib import Path
from threading import Thread
from typing import Any

import soundfile as sf  # type: ignore

from .config import SAMPLE_RATE, WAV_EXT
from .llm import Character
from .resume import get_command_dir
from .utils import prompt_choice

VOICES_FILE = "voices.json"
PRESETS_DIR = "presets"
SKIP_CHAR = "__skip__"  # sentinel: user passed on the character, not a voice


def voices_path(workdir: Path) -> Path:
    """path to the character → voice_id mapping."""
    return get_command_dir(workdir, "audition") / VOICES_FILE


def load_voices(workdir: Path) -> dict[str, str]:
    """load the character → voice_id mapping, or {} if absent."""
    p = voices_path(workdir)
    if not p.exists():
        return {}
    data: dict[str, str] = json.loads(p.read_text())
    return data


def save_voices(workdir: Path, voices: dict[str, str]) -> None:
    """persist the character → voice_id mapping.

    no-op when the mapping already on disk is identical: --step advances only
    when a phase touched files, so rewriting an unchanged voices.json pins the
    pipeline on audition and it never reaches script.
    """
    p = voices_path(workdir)
    payload = json.dumps(voices, indent=2)
    if p.exists() and p.read_text() == payload:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload)


def _play_wav_async(path: Path) -> subprocess.Popen | None:
    """spawn a non-blocking player subprocess; returns Popen handle or None.

    stdin is detached: ffplay reads the terminal for its own key bindings, and
    a player left running would swallow the keystrokes meant for our prompt.
    """
    player = shutil.which("ffplay")
    if player:
        return subprocess.Popen(
            [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    for cmd in ("paplay", "aplay", "afplay"):
        p = shutil.which(cmd)
        if p:
            return subprocess.Popen(
                [p, str(path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    print(f"(no audio player found; listen to {path} manually)")
    return None


def _play_pcm_stream(sample_rate: int) -> subprocess.Popen | None:
    """spawn ffplay reading s16le mono PCM from stdin; returns Popen with
    a writable stdin pipe, or None if ffplay isn't available.
    """
    player = shutil.which("ffplay")
    if not player:
        return None
    # ffplay >= 5.x dropped -ac; use -ch_layout mono (works on both old and new).
    return subprocess.Popen(
        [
            player,
            "-nodisp",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ch_layout",
            "mono",
            "-i",
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_playback(proc: subprocess.Popen | None) -> None:
    """terminate a running playback subprocess, if any."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()


def _preview_path(workdir: Path, char_name: str, voice_id: str) -> Path:
    """path where a character-voice audition preview wav is cached."""
    presets_dir = get_command_dir(workdir, "audition") / PRESETS_DIR
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir / f"{char_name}__{voice_id}{WAV_EXT}"


def _synthesize_preview(
    engine: Any, char: Character, voice_id: str, preview_path: Path
) -> None:
    """synthesize a preview wav of the character's audition line with voice_id."""
    if preview_path.exists():
        return
    original = engine.config.speaker
    try:
        engine.config.speaker = voice_id
        audio, sr = engine.synthesize(char.audition_text())
        sf.write(str(preview_path), audio, sr)
    finally:
        engine.config.speaker = original


def _pcm_writer(stdin: Any) -> Any:
    """feed pcm to a player, ignoring a pipe the user has already closed."""

    def write(chunk: bytes) -> None:
        try:
            stdin.write(chunk)
            stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    return write


def _stream_preview(engine: Any, text: str, voice_id: str, preview_path: Path) -> Any:
    """render a take straight into the player, caching the wav when it lands.

    returns the player handle, or None when live playback is unavailable and
    the caller should fall back to synthesizing the whole take first.
    """
    proc = _play_pcm_stream(SAMPLE_RATE)
    if proc is None or proc.stdin is None:
        return None
    stdin = proc.stdin

    def run() -> None:
        try:
            audio, sr = engine.design_voice_stream(
                text=text, instruct="", on_chunk=_pcm_writer(stdin), voice=voice_id
            )
            sf.write(str(preview_path), audio, sr)
        except Exception as e:
            print(f"    stream failed: {e}")
        finally:
            try:
                stdin.close()
            except OSError:
                pass

    Thread(target=run, daemon=True).start()
    return proc


def _start_preview(engine: Any, char: Character, voice_id: str, preview: Path) -> Any:
    """begin playing a take without blocking, so the prompt stays usable.

    a streaming backend plays as it renders; anything else renders first and
    then plays the cached wav.
    """
    if not preview.exists() and getattr(engine, "streaming", False):
        proc = _stream_preview(engine, char.audition_text(), voice_id, preview)
        if proc is not None:
            return proc
    _synthesize_preview(engine, char, voice_id, preview)
    return _play_wav_async(preview)


def _audition_voices(
    workdir: Path, char: Character, available: list[str], engine: Any
) -> tuple[str | None, bool]:
    """walk the voice list for one character; returns (choice, quit_requested).

    choice is the accepted voice id, SKIP_CHAR when the user passed on the
    character, or None when the list ran out. navigation is by index rather
    than a plain walk so [p]rev can go back, replaying the cached take.
    """
    index = 0
    while 0 <= index < len(available):
        voice_id = available[index]
        preview = _preview_path(workdir, char.name, voice_id)
        print(f"\n  trying '{voice_id}' ({index + 1}/{len(available)})...")
        try:
            playback = _start_preview(engine, char, voice_id, preview)
        except Exception as e:
            print(f"    failed to synthesize preview: {e}")
            index += 1
            continue

        step = 1
        while True:
            ans = prompt_choice(
                "  [y]es / [n]ext / [p]rev / [r]eplay / [s]kip char / [q]uit: "
            )
            # any answer cuts the take short rather than waiting it out
            _stop_playback(playback)
            if ans in ("y", "yes"):
                return voice_id, False
            if ans in ("n", "next", ""):
                break
            if ans in ("p", "prev"):
                if index == 0:
                    print("    already at the first voice")
                    continue
                step = -1
                break
            if ans in ("r", "replay"):
                # never re-synthesize: on a metered backend that is a
                # second charge for a take already paid for
                if preview.exists():
                    playback = _play_wav_async(preview)
                else:
                    print("    still rendering; replay again in a moment")
                continue
            if ans in ("s", "skip"):
                return SKIP_CHAR, False
            if ans in ("q", "quit"):
                return None, True
            print("    unknown input")
        index += step
    return None, False


def run_casting(
    workdir: Path,
    cast: list[Character],
    engine: Any,
    force: bool = False,
) -> dict[str, str]:
    """interactively assign a preset voice to each character.

    plays each voice saying the character's audition line; the user navigates
    the list with yes/next/prev/replay/skip/quit while it plays.
    resumable: existing mappings are preserved unless force=True.
    """
    available = engine.list_voices()
    if not available:
        raise RuntimeError("no preset voices returned from backend")

    print(f"casting: {len(available)} preset voices available: {', '.join(available)}")

    voices = {} if force else load_voices(workdir)
    quit_requested = False

    for char in cast:
        if quit_requested:
            break
        if char.name in voices and voices[char.name]:
            print(f"casting: {char.name} already cast as '{voices[char.name]}' (skip)")
            continue

        print(f"\n=== {char.name} ===")
        print(f"  {char.description}")
        print(f"  voice: {char.voice_prompt()}")
        print(f"  line: {char.audition_text()!r}")

        cast_voice, quit_requested = _audition_voices(workdir, char, available, engine)

        if cast_voice and cast_voice != SKIP_CHAR:
            voices[char.name] = cast_voice
            save_voices(workdir, voices)
            print(f"  cast: {char.name} -> {cast_voice}")
        elif cast_voice == SKIP_CHAR:
            print(f"  skipped: {char.name} (no voice assigned)")

    save_voices(workdir, voices)
    print(f"\ncasting: saved {len(voices)} assignments to {voices_path(workdir)}")
    return voices
