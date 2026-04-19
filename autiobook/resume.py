"""resumability utilities."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, cast

from .config import STATE_FILE


def get_command_dir(workdir: Path, command: str) -> Path:
    """get and create directory for a specific command."""
    path = workdir / command
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_chapters(
    metadata: Dict[str, Any],
    source_dir: Path,
    target_dir: Path,
    chapters_filter: list[int] | None = None,
    source_ext: str = ".txt",
    target_ext: str = ".wav",
) -> List[tuple[int, Path, Path]]:
    """list (index, source, target) for chapters based on metadata."""
    results = []
    for c in metadata.get("chapters", []):
        idx = c["index"]
        if chapters_filter and idx not in chapters_filter:
            continue

        base = c.get("filename_base")
        if not base:
            continue

        source_path = source_dir / (base + source_ext)
        target_path = target_dir / (base + target_ext)

        if source_path.exists():
            results.append((idx, source_path, target_path))

    return results


def compute_hash(obj: Any) -> str:
    """compute stable sha256 hash of a json-serializable object."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load_state(path: Path) -> Dict[str, Any]:
    """load json state file, returning empty dict if missing/invalid."""
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return cast(Dict[str, Any], json.load(f))
    except Exception:
        return {}


def save_state(path: Path, data: Dict[str, Any]):
    """save json state file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class ResumeManager:
    """helper for managing granular resume state (e.g. for chunks/segments)."""

    def __init__(self, state_path: Path, force: bool = False):
        self.state_path = state_path
        self.state = load_state(state_path)
        self.dirty = False
        self.force = force

    @classmethod
    def for_command(
        cls, workdir: Path, command: str, force: bool = False
    ) -> "ResumeManager":
        """create a resume manager for a specific command."""
        state_path = get_command_dir(workdir, command) / STATE_FILE
        return cls(state_path, force=force)

    def is_fresh(self, key: str, current_hash: str) -> bool:
        """check if key exists, matches hash, and is marked done."""
        if self.force:
            return False
        val = self.state.get(str(key))
        return (
            isinstance(val, dict)
            and val.get("hash") == current_hash
            and val.get("done", False)
        )

    def update(self, key: str, current_hash: str, **extra: Any):
        """mark key as fully complete. extra kwargs are stored alongside."""
        entry: Dict[str, Any] = {"hash": current_hash, "done": True}
        entry.update(extra)
        self.state[str(key)] = entry
        self.dirty = True

    def get_partial(self, key: str) -> Dict[str, Any] | None:
        """get partial state for key."""
        val = self.state.get(str(key))
        if isinstance(val, dict) and val.get("hash") and not val.get("done", False):
            return val
        return None

    def set_partial(self, key: str, data: Dict[str, Any]):
        """set partial state for key."""
        data["done"] = False
        self.state[str(key)] = data
        self.dirty = True

    def clear_partial(self, key: str):
        """clear state for key."""
        key_str = str(key)
        if key_str in self.state:
            del self.state[key_str]
            self.dirty = True

    def save(self):
        """save state to disk if modified."""
        if self.dirty:
            save_state(self.state_path, self.state)
            self.dirty = False
