"""tests for preview playback in the directed casting loop.

playback must not block the prompt: the user judges a voice while it plays
and moves on mid-take, rather than sitting through every audition line.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf  # type: ignore

from autiobook.casting import (
    SKIP_CHAR,
    _audition_voices,
    _start_preview,
    load_voices,
    run_casting,
    save_voices,
    voices_path,
)
from autiobook.config import SAMPLE_RATE
from autiobook.llm import Character
from autiobook.resume import get_command_dir
from autiobook.utils import dir_mtime

VOICES = ["Zephyr", "Puck", "Charon"]

TAKE = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)


def _char() -> Character:
    return Character(name="Narrator", description="d", audition_line="hello there")


class FakeEngine:
    """records how a take was requested."""

    def __init__(self, streaming: bool):
        self.streaming = streaming
        self.config = type("cfg", (), {"speaker": "default"})()
        self.streamed: list[tuple[str, str]] = []
        self.synthesized: list[str] = []

    def design_voice_stream(self, text, instruct, on_chunk=None, cancel=None, voice=""):
        self.streamed.append((text, voice))
        if on_chunk:
            on_chunk(b"\x00\x00" * 100)
        return TAKE, SAMPLE_RATE

    def synthesize(self, text, instruct="", speaker=None):
        self.synthesized.append(text)
        return TAKE, SAMPLE_RATE

    def list_voices(self):
        return list(VOICES)


def _fake_player() -> MagicMock:
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.poll.return_value = None
    return proc


class InlineThread:
    """runs the worker on start(), so the test sees its result deterministically."""

    def __init__(self, target=None, daemon=False):
        self._target = target

    def start(self) -> None:
        self._target()


class TestStartPreview:
    def test_streaming_engine_plays_while_rendering(self, tmp_path):
        engine = FakeEngine(streaming=True)
        preview = tmp_path / "Narrator__Zephyr.wav"
        player = _fake_player()

        with patch("autiobook.casting._play_pcm_stream", return_value=player):
            with patch("autiobook.casting.Thread", InlineThread):
                handle = _start_preview(engine, _char(), "Zephyr", preview)

        assert handle is player
        assert engine.synthesized == []
        assert engine.streamed == [("hello there", "Zephyr")]
        # pcm reached the player, and the take was cached for replay
        player.stdin.write.assert_called()
        assert preview.exists()

    def test_non_streaming_engine_renders_then_plays(self, tmp_path):
        engine = FakeEngine(streaming=False)
        preview = tmp_path / "Narrator__ryan.wav"
        player = _fake_player()

        with patch("autiobook.casting._play_wav_async", return_value=player) as play:
            handle = _start_preview(engine, _char(), "ryan", preview)

        assert handle is player
        assert engine.streamed == []
        assert engine.synthesized == ["hello there"]
        assert preview.exists()
        play.assert_called_once_with(preview)

    def test_cached_take_is_replayed_not_resynthesized(self, tmp_path):
        """a metered backend must not be charged twice for the same take."""
        engine = FakeEngine(streaming=True)
        preview = tmp_path / "Narrator__Zephyr.wav"
        sf.write(str(preview), TAKE, SAMPLE_RATE)

        with patch("autiobook.casting._play_wav_async", return_value=_fake_player()):
            _start_preview(engine, _char(), "Zephyr", preview)

        assert engine.streamed == []
        assert engine.synthesized == []

    def test_falls_back_when_no_live_player(self, tmp_path):
        """without ffplay there is no pcm sink, so render the whole take."""
        engine = FakeEngine(streaming=True)
        preview = tmp_path / "Narrator__Zephyr.wav"

        with patch("autiobook.casting._play_pcm_stream", return_value=None):
            with patch("autiobook.casting._play_wav_async", return_value=None):
                _start_preview(engine, _char(), "Zephyr", preview)

        assert engine.streamed == []
        assert engine.synthesized == ["hello there"]


class TestResumedRunIsInert:
    """--step advances only when a phase touched files, so a fully cast run
    must leave audition/ alone -- otherwise the pipeline stops on audition
    forever and never reaches script."""

    def test_save_voices_is_idempotent(self, tmp_path):
        save_voices(tmp_path, {"Narrator": "Zephyr"})
        path = voices_path(tmp_path)
        before = path.stat().st_mtime_ns

        save_voices(tmp_path, {"Narrator": "Zephyr"})

        assert path.stat().st_mtime_ns == before

    def test_save_voices_writes_a_change(self, tmp_path):
        save_voices(tmp_path, {"Narrator": "Zephyr"})
        path = voices_path(tmp_path)
        before = path.stat().st_mtime_ns

        save_voices(tmp_path, {"Narrator": "Puck"})

        assert path.stat().st_mtime_ns != before
        assert load_voices(tmp_path) == {"Narrator": "Puck"}

    def test_fully_cast_run_touches_nothing(self, tmp_path):
        engine = FakeEngine(streaming=False)
        cast = [_char()]
        save_voices(tmp_path, {"Narrator": "Zephyr"})
        audition_dir = get_command_dir(tmp_path, "audition")
        before = dir_mtime(audition_dir)

        assert run_casting(tmp_path, cast, engine) == {"Narrator": "Zephyr"}
        assert dir_mtime(audition_dir) == before


class TestVoiceNavigation:
    """walking the preset voice list for one character."""

    def _run(self, tmp_path, answers: list[str]):
        """drive the loop with scripted keypresses; returns (result, played)."""
        engine = FakeEngine(streaming=False)
        played: list[str] = []

        def start_preview(_engine, _char, voice_id, _preview):
            played.append(voice_id)
            return None

        with patch("autiobook.casting._prompt", side_effect=answers):
            with patch("autiobook.casting._start_preview", start_preview):
                with patch("autiobook.casting._stop_playback"):
                    result = _audition_voices(tmp_path, _char(), VOICES, engine)
        return result, played

    def test_next_walks_forward(self, tmp_path):
        (choice, quit_requested), played = self._run(tmp_path, ["n", "n", "y"])
        assert choice == "Charon"
        assert quit_requested is False
        assert played == VOICES

    def test_prev_goes_back_to_an_earlier_voice(self, tmp_path):
        (choice, _), played = self._run(tmp_path, ["n", "n", "p", "y"])
        assert choice == "Puck"
        assert played == ["Zephyr", "Puck", "Charon", "Puck"]

    def test_prev_at_the_first_voice_stays_put(self, tmp_path):
        (choice, _), played = self._run(tmp_path, ["p", "y"])
        assert choice == "Zephyr"
        assert played == ["Zephyr"]

    def test_skip_character(self, tmp_path):
        (choice, quit_requested), _ = self._run(tmp_path, ["s"])
        assert choice == SKIP_CHAR
        assert quit_requested is False

    def test_quit(self, tmp_path):
        (choice, quit_requested), _ = self._run(tmp_path, ["n", "q"])
        assert choice is None
        assert quit_requested is True

    def test_exhausting_the_list_returns_no_choice(self, tmp_path):
        (choice, quit_requested), played = self._run(tmp_path, ["n", "n", "n"])
        assert choice is None
        assert quit_requested is False
        assert played == VOICES
