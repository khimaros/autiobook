"""tests for retake: corrupted segment detection and removal."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore

SAMPLE_RATE = 24000


def _write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), SAMPLE_RATE)


def _good(duration_s: float = 1.0) -> np.ndarray:
    """speech-like: tone bursts separated by silence (gaps → high crest, low median)."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    tone = 0.4 * np.sin(2 * np.pi * 220 * t)
    # burst envelope: ~30% active, rest silent — typical speech duty cycle
    period = max(1, n // 8)
    env = np.zeros(n, dtype=np.float32)
    for start in range(0, n, period):
        end = min(start + period // 3, n)
        ramp = np.sin(np.pi * np.linspace(0, 1, end - start))
        env[start:end] = ramp
    a = (env * tone).astype(np.float32)
    a[0] = 0.0  # avoid click at start
    a[-1] = 0.0  # avoid truncated at end
    return a


class TestDetection:
    def test_silent(self, tmp_path):
        from autiobook.retake import analyze_segment

        p = tmp_path / "silent.wav"
        _write(p, np.zeros(SAMPLE_RATE))
        m = analyze_segment(p)
        assert "silent" in m.categories

    def test_click_at_start(self, tmp_path):
        from autiobook.retake import analyze_segment

        p = tmp_path / "click.wav"
        a = _good()
        a[0] = 0.8  # discontinuity at first sample
        _write(p, a)
        m = analyze_segment(p)
        assert "click" in m.categories

    def test_truncated_end(self, tmp_path):
        from autiobook.retake import analyze_segment

        p = tmp_path / "trunc.wav"
        a = _good()
        a[-1] = -0.6
        _write(p, a)
        m = analyze_segment(p)
        assert "truncated" in m.categories

    def test_clipping(self, tmp_path):
        from autiobook.retake import analyze_segment

        p = tmp_path / "clip.wav"
        a = _good()
        a[100:200] = 1.0  # 100 saturated samples
        _write(p, a)
        m = analyze_segment(p)
        assert "clipping" in m.categories

    def test_clean_segment_has_no_categories(self, tmp_path):
        from autiobook.retake import analyze_segment

        p = tmp_path / "ok.wav"
        _write(p, _good())
        m = analyze_segment(p)
        assert m.categories == []


def _flat_loud(duration_s: float) -> np.ndarray:
    """flat, loud-throughout noise — fires the tight noisy rule at any duration."""
    n = int(SAMPLE_RATE * duration_s)
    rng = np.random.default_rng(0)
    return rng.uniform(-0.5, 0.5, size=n).astype(np.float32)


def _loose_noisy(duration_s: float) -> np.ndarray:
    """metrics that only satisfy the loose noisy rule (crest~4, med~0.075)."""
    n = int(SAMPLE_RATE * duration_s)
    rng = np.random.default_rng(1)
    signs = rng.choice([-1.0, 1.0], size=n)
    base = 0.075 * signs
    k = max(1, n // 100)
    idx = rng.choice(n, size=k, replace=False)
    base[idx] = 0.3 * rng.choice([-1.0, 1.0], size=k)
    # zero the endpoints so click/truncated rules don't fire on the synthetic
    # noise floor; the test targets the "noisy" classifier alone.
    base[0] = 0.0
    base[-1] = 0.0
    return base.astype(np.float32)


class TestNoisy:
    def test_tight_fires_regardless_of_duration(self):
        from autiobook.retake import categorize_audio

        assert "noisy" in categorize_audio(_flat_loud(0.5))
        assert "noisy" in categorize_audio(_flat_loud(4.0))

    def test_loose_requires_minimum_duration(self):
        from autiobook.retake import categorize_audio

        assert "noisy" not in categorize_audio(_loose_noisy(1.0))
        assert "noisy" in categorize_audio(_loose_noisy(4.0))

    def test_clean_speech_like_not_flagged(self):
        from autiobook.retake import categorize_audio

        assert "noisy" not in categorize_audio(_good(4.0))


class TestCommand:
    def _setup(self, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
        """workdir with one good and one silent segment, plus timing+script."""
        workdir = tmp_path
        perform = workdir / "perform"
        segments = perform / "segments"
        segments.mkdir(parents=True)

        good = segments / "aaaa.wav"
        bad = segments / "bbbb.wav"
        _write(good, _good())
        _write(bad, np.zeros(SAMPLE_RATE))

        script_dir = workdir / "script"
        script_dir.mkdir()
        script = script_dir / "01_ch.json"
        script.write_text(
            json.dumps(
                {
                    "version": 2,
                    "segments": [
                        {
                            "speaker": "Narrator",
                            "text": "hello world",
                            "instruction": "neutral",
                        },
                        {
                            "speaker": "Narrator",
                            "text": "the silent one",
                            "instruction": "neutral",
                        },
                    ],
                }
            )
        )

        timing = perform / "01_ch.wav.timing.json"
        timing.write_text(
            json.dumps(
                {
                    "chunks": [
                        {
                            "hash": "aaaa",
                            "start_s": 0.0,
                            "end_s": 1.0,
                            "script_idx": 0,
                            "chunk_idx": 0,
                            "script_path": str(script),
                        },
                        {
                            "hash": "bbbb",
                            "start_s": 1.5,
                            "end_s": 2.5,
                            "script_idx": 1,
                            "chunk_idx": 0,
                            "script_path": str(script),
                        },
                    ]
                }
            )
        )
        return workdir, segments, good, bad

    def test_dry_run_preserves_files(self, tmp_path, capsys):
        from argparse import Namespace

        from autiobook.retake import cmd_retake

        workdir, _, good, bad = self._setup(tmp_path)
        cmd_retake(
            Namespace(
                workdir=str(workdir), command="perform", dry_run=True, verbose=False
            )
        )

        out = capsys.readouterr().out
        assert "silent" in out
        assert "the silent one" in out  # text lookup worked
        assert good.exists()
        assert bad.exists()

    def test_delete_removes_offenders_only(self, tmp_path, monkeypatch, capsys):
        from argparse import Namespace

        from autiobook import retake as retake_mod

        captured: dict = {}

        def fake_regen(workdir, cmd, chapters, config, verbose, hashes):
            captured["hashes"] = hashes

        monkeypatch.setattr(retake_mod, "_regenerate", fake_regen)

        workdir, _, good, bad = self._setup(tmp_path)
        retake_mod.cmd_retake(
            Namespace(
                workdir=str(workdir), command="perform", dry_run=False, verbose=False
            )
        )

        assert good.exists()
        assert not bad.exists()
        # regeneration should be scoped to the deleted wav's hash
        assert captured["hashes"] == {"bbbb"}

    def test_prune_skips_regen(self, tmp_path, monkeypatch):
        from argparse import Namespace

        from autiobook import retake as retake_mod

        called = {"n": 0}
        monkeypatch.setattr(
            retake_mod,
            "_regenerate",
            lambda *a, **kw: called.__setitem__("n", called["n"] + 1),
        )

        workdir, _, good, bad = self._setup(tmp_path)
        retake_mod.cmd_retake(
            Namespace(
                workdir=str(workdir),
                command="perform",
                dry_run=False,
                verbose=False,
                prune=True,
            )
        )

        assert not bad.exists()
        assert called["n"] == 0

    def test_no_offenders(self, tmp_path, capsys):
        from argparse import Namespace

        from autiobook.retake import cmd_retake

        perform = tmp_path / "perform"
        segments = perform / "segments"
        segments.mkdir(parents=True)
        _write(segments / "aaaa.wav", _good())

        cmd_retake(Namespace(workdir=str(tmp_path), command="perform", dry_run=False))
        out = capsys.readouterr().out
        assert "no offenders found" in out

    def test_missing_segments_dir(self, tmp_path, capsys):
        from argparse import Namespace

        from autiobook.retake import cmd_retake

        cmd_retake(Namespace(workdir=str(tmp_path), command="perform", dry_run=False))
        out = capsys.readouterr().out
        assert "no segments directory" in out
