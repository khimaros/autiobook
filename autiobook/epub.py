"""epub parsing and chapter extraction."""

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import ebooklib  # type: ignore
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning  # type: ignore
from ebooklib import epub  # type: ignore

from .config import (
    CONTENT_TAGS,
    COVER_FILE,
    METADATA_FILE,
    MIN_CHAPTER_WORDS,
    SKIP_TAGS,
    TXT_EXT,
)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


@dataclass
class Chapter:
    """single chapter extracted from an epub."""

    index: int
    title: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def filename_base(self) -> str:
        """sanitized filename without extension."""
        from .config import UNSAFE_FILENAME_CHARS

        safe_title = UNSAFE_FILENAME_CHARS.sub("_", self.title)
        safe_title = safe_title.strip().replace(" ", "_")[:50]
        return f"{self.index:02d}_{safe_title}"


@dataclass
class Book:
    """parsed epub book with metadata and chapters."""

    title: str
    author: str
    language: str
    chapters: list[Chapter]

    def to_metadata(self) -> dict:
        """return metadata dict for serialization (without chapter text)."""
        return {
            "title": self.title,
            "author": self.author,
            "language": self.language,
            "chapters": [
                {"index": c.index, "title": c.title, "filename_base": c.filename_base}
                for c in self.chapters
            ],
        }


def extract_text_from_html(html_content: bytes) -> str:
    """convert html content to clean plain text."""
    soup = BeautifulSoup(html_content, "lxml")

    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()

    paragraphs = []
    for tag in soup.find_all(CONTENT_TAGS):
        if tag.find(CONTENT_TAGS):
            # container with nested content tags
            # extract text from non-content children to avoid skipping mixed content
            parts = []
            for child in tag.children:
                # child.name is None for NavigableString (text nodes)
                name = getattr(child, "name", None)
                if name is None or name not in CONTENT_TAGS:
                    parts.append(child.get_text())

            text = " ".join("".join(parts).split())
            if text:
                paragraphs.append(text)
            continue

        text = " ".join(tag.get_text().split())
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def extract_title_from_html(html_content: bytes) -> str | None:
    """extract chapter title from html."""
    soup = BeautifulSoup(html_content, "lxml")

    # try title tag first
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        return title_tag.string.strip()

    # try first heading
    for tag in ["h1", "h2", "h3"]:
        heading = soup.find(tag)
        if heading:
            text = heading.get_text(strip=True)
            if text:
                return text

    return None


def extract_cover(book: epub.EpubBook) -> bytes | None:
    """extract cover image from epub, returns image bytes or None."""
    # 1. try common IDs
    for cid in ["cover", "cover-image", "coverimage"]:
        if item := book.get_item_with_id(cid):
            return cast(bytes, item.get_content())

    # 2. try OPF cover metadata
    if cover_meta := book.get_metadata("OPF", "cover"):
        if item := book.get_item_with_id(str(cover_meta[0][0])):
            return cast(bytes, item.get_content())

    # 3. search images
    images = list(book.get_items_of_type(ebooklib.ITEM_IMAGE))
    for item in images:
        if "cover" in item.get_name().lower():
            return cast(bytes, item.get_content())

    return cast(bytes, images[0].get_content()) if images else None


def parse_epub(path: Path) -> tuple[Book, bytes | None]:
    """parse an epub file and extract all chapters with metadata and cover."""
    eb = epub.read_epub(str(path), options={"ignore_ncx": True})

    def get_meta(key, default=""):
        m = eb.get_metadata("DC", key)
        return str(m[0][0]) if m and m[0] else default

    book = Book(
        title=get_meta("title", path.stem),
        author=get_meta("creator", "Unknown"),
        language=get_meta("language", "en"),
        chapters=[],
    )

    for item in eb.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = cast(bytes, item.get_content())
        text = extract_text_from_html(content)
        if len(text.split()) < MIN_CHAPTER_WORDS:
            continue

        idx = len(book.chapters) + 1
        title = extract_title_from_html(content) or f"Chapter {idx}"
        book.chapters.append(Chapter(index=idx, title=title, text=text))

    return book, extract_cover(eb)


def ensure_extracted(epub_path: Path, workdir: Path, force: bool = False) -> None:
    """ensure epub is extracted to workdir, skipping if already fresh."""
    from .resume import get_command_dir

    if not epub_path.exists():
        msg = f"epub file not found: {epub_path}"
        print(msg)
        raise FileNotFoundError(msg)

    extract_dir = get_command_dir(workdir, "extract")
    state_path = extract_dir / "state.json"

    with open(epub_path, "rb") as f:
        epub_hash = hashlib.sha256(f.read()).hexdigest()

    if not force and state_path.exists():
        try:
            with open(state_path) as f:
                if json.load(f).get("epub_hash") == epub_hash:
                    if any(extract_dir.glob(f"*{TXT_EXT}")):
                        return
        except Exception:
            pass

    book, cover_data = parse_epub(epub_path)
    save_extracted(book, workdir, cover_data)

    with open(state_path, "w") as f:
        json.dump({"epub_hash": epub_hash}, f, indent=2)


def save_extracted(book: Book, workdir: Path, cover_data: bytes | None = None) -> None:
    """save extracted chapters and cover to workdir."""
    from .resume import get_command_dir

    extract_dir = get_command_dir(workdir, "extract")

    # save metadata
    metadata_path = extract_dir / METADATA_FILE
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(book.to_metadata(), f, indent=2)

    # save cover image
    if cover_data:
        cover_path = extract_dir / COVER_FILE
        with open(cover_path, "wb") as f:
            f.write(cover_data)
        print(f"  saved cover image ({len(cover_data)} bytes)")

    # save chapter text files
    for chapter in book.chapters:
        txt_path = extract_dir / f"{chapter.filename_base}{TXT_EXT}"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(chapter.text)


def load_metadata(workdir: Path) -> dict:
    """load metadata from workdir."""
    from .config import UNSAFE_FILENAME_CHARS
    from .resume import get_command_dir

    metadata_path = get_command_dir(workdir, "extract") / METADATA_FILE
    with open(metadata_path, encoding="utf-8") as f:
        data = cast(dict, json.load(f))

    # ensure filename_base exists for all chapters (backwards compatibility)
    for c in data.get("chapters", []):
        if "filename_base" not in c:
            idx = c["index"]
            title = c.get("title", f"Chapter {idx}")
            safe_title = UNSAFE_FILENAME_CHARS.sub("_", title)
            safe_title = safe_title.strip().replace(" ", "_")[:50]
            c["filename_base"] = f"{idx:02d}_{safe_title}"

    return data
