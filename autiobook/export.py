"""mp3 export with id3 metadata."""

from dataclasses import dataclass
from pathlib import Path

from mutagen.id3 import APIC, ID3
from mutagen.mp3 import MP3
from pydub import AudioSegment

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
) -> int:
    """export all chapters as mp3 files with cover art."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(workdir)
    book_title = metadata["title"]
    author = metadata["author"]
    chapters = metadata["chapters"]
    total_tracks = len(chapters)

    # check for cover image
    cover_path = workdir / COVER_FILE
    if not cover_path.exists():
        cover_path = None

    newly_exported_count = 0
    skipped_count = 0
    chapters_to_process = 0

    for chapter_info in chapters:
        filename_base = chapter_info["filename_base"]
        wav_path = workdir / f"{filename_base}{WAV_EXT}"
        mp3_path = output_dir / f"{filename_base}{MP3_EXT}"

        # skip if wav doesn't exist (not synthesized yet)
        if not wav_path.exists():
            continue
        chapters_to_process += 1

        # skip if already exported (idempotent)
        if not force and mp3_path.exists():
            skipped_count += 1
            continue

        print(f"exporting {wav_path.name}...")

        mp3_meta = MP3Metadata(
            title=chapter_info["title"],
            album=book_title,
            artist=author,
            track_number=chapter_info["index"],
            total_tracks=total_tracks,
        )

        wav_to_mp3(wav_path, mp3_path, mp3_meta, bitrate, cover_path)
        print(f"  -> {mp3_path.name}")
        newly_exported_count += 1

    if newly_exported_count == 0 and skipped_count == 0 and chapters_to_process == 0:
        print("export: no chapters to export.")
    elif newly_exported_count == 0 and skipped_count > 0:
        print(f"export: all {skipped_count} chapters up to date.")

    return newly_exported_count + skipped_count
