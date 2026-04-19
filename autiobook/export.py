"""mp3 export with id3 metadata."""

import contextlib
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from mutagen.id3 import APIC, ID3  # type: ignore
from mutagen.mp3 import MP3  # type: ignore
from pydub import AudioSegment  # type: ignore

from .config import COVER_FILE, DEFAULT_BITRATE, M4B_EXT, MP3_EXT, WAV_EXT
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


def get_wav_duration_ms(path: Path) -> int:
    """get duration of wav file in milliseconds."""
    with contextlib.closing(wave.open(str(path), "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return int((frames / rate) * 1000)


def escape_ffmetadata(text: str) -> str:
    """escape special characters for ffmetadata format."""
    chars = ["=", ";", "#", "\\", "\n"]
    for c in chars:
        text = text.replace(c, "\\" + c)
    return text


def export_m4b(
    chapters: list[tuple[int, Path, Path]],
    info_map: dict,
    meta: dict,
    output_dir: Path,
    cover_path: Path | None,
    bitrate: str,
) -> tuple[int, int]:
    """export chapters as single m4b file with metadata."""
    # 1. calculate durations and build metadata
    ffmetadata = [";FFMETADATA1"]

    # global metadata
    global_meta = {
        "title": meta["title"],
        "artist": meta["author"],
        "album": meta["title"],
        "genre": "Audiobook",
    }
    for k, v in global_meta.items():
        if v:
            ffmetadata.append(f"{k}={escape_ffmetadata(str(v))}")

    # chapters
    current_time = 0
    file_list = []

    # sort chapters by index
    sorted_chapters = sorted(chapters, key=lambda x: x[0])

    for idx, wav_path, _ in sorted_chapters:
        duration = get_wav_duration_ms(wav_path)
        c_info = info_map.get(idx)
        title = c_info["title"] if c_info else f"Chapter {idx}"

        ffmetadata.append("[CHAPTER]")
        ffmetadata.append("TIMEBASE=1/1000")
        ffmetadata.append(f"START={current_time}")
        ffmetadata.append(f"END={current_time + duration}")
        ffmetadata.append(f"title={escape_ffmetadata(title)}")

        file_list.append(f"file '{wav_path.absolute()}'")
        current_time += duration

    # write temp files
    meta_path = output_dir / "metadata.txt"
    list_path = output_dir / "files.txt"

    meta_path.write_text("\n".join(ffmetadata), encoding="utf-8")
    list_path.write_text("\n".join(file_list), encoding="utf-8")

    # output file
    # sanitize filename
    safe_title = "".join(
        c for c in meta["title"] if c.isalnum() or c in (" ", "-", "_")
    ).strip()
    output_file = output_dir / f"{safe_title}{M4B_EXT}"

    # build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
    ]

    input_idx = 1
    map_audio = "0:a"
    map_video = None

    if cover_path:
        cmd.extend(["-i", str(cover_path)])
        map_video = f"{input_idx}:v"
        input_idx += 1

    cmd.extend(["-i", str(meta_path)])
    map_meta = input_idx

    cmd.extend(["-map_metadata", str(map_meta)])
    cmd.extend(["-map", map_audio])

    if map_video:
        cmd.extend(["-map", map_video])
        cmd.extend(["-c:v", "copy", "-disposition:v:0", "attached_pic"])

    cmd.extend(["-c:a", "aac", "-b:a", bitrate])
    cmd.extend([str(output_file)])

    print(f"export: creating m4b file {output_file.name}...")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"export: ffmpeg failed: {e.stderr.decode()}")
        raise

    # clean up temp files
    meta_path.unlink()
    list_path.unlink()

    return len(chapters), 0


def export_audiobook(
    workdir: Path,
    output_dir: Path,
    bitrate: str = DEFAULT_BITRATE,
    force: bool = False,
    m4b: bool = False,
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

    new, skipped = 0, 0
    info_map = {c["index"]: c for c in meta["chapters"]}

    if m4b:
        return export_m4b(
            chapter_paths, info_map, meta, output_dir, cover_path, bitrate
        )

    resume = ResumeManager.for_command(workdir, "export", force=force)

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
