"""interactive casting: assign preset voices to characters one take at a time."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf  # type: ignore

from .config import WAV_EXT
from .llm import Character
from .resume import get_command_dir

VOICES_FILE = "voices.json"
PRESETS_DIR = "presets"


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
    """persist the character → voice_id mapping."""
    p = voices_path(workdir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(voices, indent=2))


def _play_wav(path: Path) -> None:
    """play a wav file via ffplay, aplay, or paplay (whichever is available)."""
    player = shutil.which("ffplay")
    if player:
        subprocess.run(
            [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
            check=False,
        )
        return
    for cmd in ("paplay", "aplay", "afplay"):
        p = shutil.which(cmd)
        if p:
            subprocess.run([p, str(path)], check=False)
            return
    print(f"(no audio player found; listen to {path} manually)")


def _play_wav_async(path: Path) -> subprocess.Popen | None:
    """spawn a non-blocking player subprocess; returns Popen handle or None."""
    player = shutil.which("ffplay")
    if player:
        return subprocess.Popen(
            [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    for cmd in ("paplay", "aplay", "afplay"):
        p = shutil.which(cmd)
        if p:
            return subprocess.Popen(
                [p, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    print(f"(no audio player found; listen to {path} manually)")
    return None


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
        audio, sr = engine.synthesize(char.audition_line)
        sf.write(str(preview_path), audio, sr)
    finally:
        engine.config.speaker = original


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


def run_casting(
    workdir: Path,
    cast: list[Character],
    engine: Any,
    force: bool = False,
) -> dict[str, str]:
    """interactively assign a preset voice to each character.

    plays each voice saying the character's audition line; user says yes/next/skip/quit.
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
        print(f"  line: {char.audition_line!r}")

        cast_voice = None
        for voice_id in available:
            preview = _preview_path(workdir, char.name, voice_id)
            print(f"\n  trying '{voice_id}'...")
            try:
                _synthesize_preview(engine, char, voice_id, preview)
            except Exception as e:
                print(f"    failed to synthesize preview: {e}")
                continue
            _play_wav(preview)

            while True:
                ans = _prompt("  [y]es / [n]ext / [r]eplay / [s]kip char / [q]uit: ")
                if ans in ("y", "yes"):
                    cast_voice = voice_id
                    break
                if ans in ("n", "next", ""):
                    break
                if ans in ("r", "replay"):
                    _play_wav(preview)
                    continue
                if ans in ("s", "skip"):
                    cast_voice = "__skip__"
                    break
                if ans in ("q", "quit"):
                    quit_requested = True
                    break
                print("    unknown input")

            if cast_voice or quit_requested:
                break

        if cast_voice and cast_voice != "__skip__":
            voices[char.name] = cast_voice
            save_voices(workdir, voices)
            print(f"  cast: {char.name} -> {cast_voice}")
        elif cast_voice == "__skip__":
            print(f"  skipped: {char.name} (no voice assigned)")

    save_voices(workdir, voices)
    print(f"\ncasting: saved {len(voices)} assignments to {voices_path(workdir)}")
    return voices
