"""shared pooling logic for batch processing."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from .audio import (
    check_segment_exists,
    save_segment,
)
from .config import SAMPLE_RATE


@dataclass
class AudioTask:
    """represents a single unit of audio generation work."""

    text: str
    segment_hash: str
    segments_dir: Path
    voice_ref_audio: Optional[Path] = None
    voice_ref_text: Optional[str] = None
    instruct: str = ""


def process_pooled_tasks(
    engine: Any,
    tasks: list[AudioTask],
    desc: str = "synthesizing",
    force: bool = False,
) -> None:
    """process a list of audio tasks with pooling by voice reference and deduplication."""
    # 1. Deduplicate tasks by hash
    unique_tasks = {}
    for task in tasks:
        if task.segment_hash not in unique_tasks:
            if force or not check_segment_exists(task.segments_dir, task.segment_hash):
                unique_tasks[task.segment_hash] = task

    if not unique_tasks:
        return

    # 2. Group by voice reference
    tasks_by_voice = defaultdict(list)
    for task in unique_tasks.values():
        key = (
            str(task.voice_ref_audio) if task.voice_ref_audio else None,
            task.voice_ref_text,
        )
        tasks_by_voice[key].append(task)

    total_items = len(unique_tasks)

    with tqdm(total=total_items, desc=desc, unit="seg") as pbar:
        for _, voice_tasks in tasks_by_voice.items():
            if not voice_tasks:
                continue

            # Process in batches
            for i in range(0, len(voice_tasks), engine.config.batch_size):
                batch = voice_tasks[i : i + engine.config.batch_size]
                texts = [t.text for t in batch]

                ref_audio = batch[0].voice_ref_audio
                ref_text = batch[0].voice_ref_text
                instruct = batch[0].instruct

                try:
                    if ref_audio:
                        wavs, _ = engine.clone_voice(texts, ref_audio, ref_text)
                    else:
                        wavs, _ = engine.synthesize(texts, instruct)

                    if isinstance(wavs, np.ndarray):
                        wavs = [wavs]

                    for j, audio in enumerate(wavs):
                        if j < len(batch):
                            save_segment(
                                batch[j].segments_dir,
                                batch[j].segment_hash,
                                audio,
                                SAMPLE_RATE,
                            )
                        pbar.update(1)

                except Exception as e:
                    print(f"batch generation failed: {e}")
                    # Fallback to silence for now to allow progress
                    silence = np.zeros(1, dtype=np.float32)
                    for _ in batch:
                        pbar.update(1)
