"""tests for seed resolution and persistence.

a run without an explicit seed picks a random one and records it in the
workdir, so resuming the same book reuses it rather than drifting to a new
seed on every invocation.
"""

import json

import pytest

from autiobook.config import SEED_FILE, active_seed, set_active_seed
from autiobook.utils import resolve_seed


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("AUTIOBOOK_SEED", raising=False)


def stored_seed(workdir):
    return json.loads((workdir / SEED_FILE).read_text())["seed"]


class TestResolveSeed:
    def test_first_run_records_a_seed(self, tmp_path):
        seed, origin = resolve_seed(tmp_path, None)
        assert origin == "random"
        assert stored_seed(tmp_path) == seed

    def test_second_run_reuses_the_recorded_seed(self, tmp_path):
        """the point of the feature: resuming a book is idempotent."""
        first, _ = resolve_seed(tmp_path, None)
        second, origin = resolve_seed(tmp_path, None)
        assert second == first
        assert origin == "stored"

    def test_flag_overrides_and_is_recorded(self, tmp_path):
        resolve_seed(tmp_path, None)
        seed, origin = resolve_seed(tmp_path, 4242)
        assert (seed, origin) == (4242, "--seed")
        assert stored_seed(tmp_path) == 4242

    def test_flag_persists_for_later_runs(self, tmp_path):
        resolve_seed(tmp_path, 4242)
        seed, origin = resolve_seed(tmp_path, None)
        assert (seed, origin) == (4242, "stored")

    def test_env_is_used_when_no_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTIOBOOK_SEED", "777")
        seed, origin = resolve_seed(tmp_path, None)
        assert (seed, origin) == (777, "env")
        assert stored_seed(tmp_path) == 777

    def test_flag_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUTIOBOOK_SEED", "777")
        seed, origin = resolve_seed(tmp_path, 4242)
        assert (seed, origin) == (4242, "--seed")

    def test_stored_zero_is_honoured(self, tmp_path):
        """0 disables seeding and must survive the round trip."""
        resolve_seed(tmp_path, 0)
        seed, origin = resolve_seed(tmp_path, None)
        assert (seed, origin) == (0, "stored")

    def test_no_workdir_does_not_persist(self, tmp_path):
        seed, origin = resolve_seed(None, 99)
        assert (seed, origin) == (99, "--seed")
        assert not (tmp_path / SEED_FILE).exists()

    def test_workdir_is_created_if_absent(self, tmp_path):
        wd = tmp_path / "new_output"
        seed, _ = resolve_seed(wd, 5)
        assert stored_seed(wd) == 5

    def test_malformed_seed_file_falls_back(self, tmp_path):
        (tmp_path / SEED_FILE).write_text("not json{")
        seed, origin = resolve_seed(tmp_path, None)
        assert origin == "random"
        assert stored_seed(tmp_path) == seed


class TestActiveSeed:
    def test_set_active_seed_is_visible_to_consumers(self):
        """tts/llm configs read the seed lazily, after args are parsed."""
        original = active_seed()
        try:
            set_active_seed(31337)
            assert active_seed() == 31337

            from autiobook.tts_http import HTTPTTSConfig

            assert HTTPTTSConfig(api_base="x", model="m").seed == 31337
        finally:
            set_active_seed(original)
