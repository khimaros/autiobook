"""mp3 export with id3 metadata."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from mutagen.id3 import APIC, ID3  # type: ignore
from mutagen.mp3 import MP3  # type: ignore
from pydub import AudioSegment  # type: ignore

from .config import COVER_FILE, DEFAULT_BITRATE, MP3_EXT, WAV_EXT
from .epub import load_metadata


@dataclass
class MP3Metadata:
    """id3 tag metadata for mp3 file."""

    title: str
    album: str
    artist: str
    track_number: int
    total_tracks: int


def wav_to_mp3(
    wav_path: Path,
    mp3_path: Path,
    metadata: MP3Metadata,
    bitrate: str = DEFAULT_BITRATE,
    cover_path: Path | None = None,
) -> None:
    """convert wav to mp3 with id3 metadata tags and cover art."""
    audio = AudioSegment.from_wav(str(wav_path))

    tags = {
        "title": metadata.title,
        "album": metadata.album,
        "artist": metadata.artist,
        "track": f"{metadata.track_number}/{metadata.total_tracks}",
    }

    audio.export(str(mp3_path), format="mp3", bitrate=bitrate, tags=tags)

    # add cover art if available
    if cover_path and cover_path.exists():
        mp3 = MP3(str(mp3_path), ID3=ID3)
        if mp3.tags is None:
            mp3.add_tags()

        if mp3.tags is not None:
            cover_data = cover_path.read_bytes()
            mp3.tags.add(
                APIC(
                    encoding=3,  # utf-8
                    mime="image/jpeg",
                    type=3,  # front cover
                    desc="Cover",
                    data=cover_data,
                )
            )
            mp3.save()


def export_audiobook(
    workdir: Path,
    output_dir: Path,
    bitrate: str = DEFAULT_BITRATE,
    force: bool = False,
) -> tuple[int, int]:
    """export all chapters as mp3 files with cover art."""
    from .resume import ResumeManager, compute_hash, get_command_dir, list_chapters

    output_dir.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(workdir)

    # find first available source dir
    source_dir = next(
        (
            d
            for d in [workdir / "perform", workdir / "synthesize"]
            if d.exists() and list_chapters(meta, d, output_dir, source_ext=WAV_EXT)
        ),
        None,
    )

    if not source_dir:
        print("export: no wav files found")
        return 0, 0

    chapter_paths = list_chapters(
        meta, source_dir, output_dir, source_ext=WAV_EXT, target_ext=MP3_EXT
    )

    cover = get_command_dir(workdir, "extract") / COVER_FILE
    cover_path = cover if cover.exists() else None

    if cover_path:
        shutil.copy(cover_path, output_dir / COVER_FILE)

    resume = ResumeManager.for_command(workdir, "export", force=force)
    new, skipped = 0, 0
    info_map = {c["index"]: c for c in meta["chapters"]}

    for idx, wav_p, mp3_p in chapter_paths:
        c_info = info_map.get(idx)
        if not c_info:
            continue
        h = compute_hash(
            {
                "size": wav_p.stat().st_size,
                "mtime": wav_p.stat().st_mtime,
                "title": c_info["title"],
                "album": meta["title"],
                "artist": meta["author"],
                "track": idx,
                "bitrate": bitrate,
                "cover": str(cover_path) if cover_path else None,
            }
        )

        if not force and mp3_p.exists() and resume.is_fresh(str(mp3_p), h):
            skipped += 1
            continue

        print(f"exporting {wav_p.name}...")
        wav_to_mp3(
            wav_p,
            mp3_p,
            MP3Metadata(
                c_info["title"],
                meta["title"],
                meta["author"],
                idx,
                len(meta["chapters"]),
            ),
            bitrate,
            cover_path,
        )
        resume.update(str(mp3_p), h)
        resume.save()
        new += 1

    return new, skipped
