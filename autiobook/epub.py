"""epub parsing and chapter extraction."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub

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
    # try common cover item ids
    for cover_id in ["cover", "cover-image", "coverimage"]:
        item = book.get_item_with_id(cover_id)
        if item and item.get_content():
            return item.get_content()

    # try cover metadata reference
    cover_meta = book.get_metadata("OPF", "cover")
    if cover_meta:
        cover_id = cover_meta[0][0] if cover_meta[0] else None
        if cover_id:
            item = book.get_item_with_id(cover_id)
            if item and item.get_content():
                return item.get_content()

    # fallback: find first image item with 'cover' in name
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if "cover" in item.get_name().lower():
            return item.get_content()

    # last resort: first image
    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        return item.get_content()

    return None


def parse_epub(path: Path) -> tuple[Book, bytes | None]:
    """parse an epub file and extract all chapters with metadata and cover."""
    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    cover_data = extract_cover(book)

    # extract metadata
    title = book.get_metadata("DC", "title")
    title = title[0][0] if title else path.stem

    author = book.get_metadata("DC", "creator")
    author = author[0][0] if author else "Unknown"

    language = book.get_metadata("DC", "language")
    language = language[0][0] if language else "en"

    # extract chapters from spine order
    chapters = []
    index = 1

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content()
        text = extract_text_from_html(content)

        # skip short/empty chapters
        if len(text.split()) < MIN_CHAPTER_WORDS:
            continue

        chapter_title = extract_title_from_html(content)
        if not chapter_title:
            chapter_title = f"Chapter {index}"

        chapters.append(Chapter(index=index, title=chapter_title, text=text))
        index += 1

    return (
        Book(title=title, author=author, language=language, chapters=chapters),
        cover_data,
    )


def save_extracted(book: Book, workdir: Path, cover_data: bytes | None = None) -> None:
    """save extracted chapters and cover to workdir."""
    workdir.mkdir(parents=True, exist_ok=True)

    # save metadata
    metadata_path = workdir / METADATA_FILE
    with open(metadata_path, "w") as f:
        json.dump(book.to_metadata(), f, indent=2)

    # save cover image
    if cover_data:
        cover_path = workdir / COVER_FILE
        with open(cover_path, "wb") as f:
            f.write(cover_data)
        print(f"  saved cover image ({len(cover_data)} bytes)")

    # save chapter text files
    for chapter in book.chapters:
        txt_path = workdir / f"{chapter.filename_base}{TXT_EXT}"
        with open(txt_path, "w") as f:
            f.write(chapter.text)


def load_metadata(workdir: Path) -> dict:
    """load metadata from workdir."""
    metadata_path = workdir / METADATA_FILE
    with open(metadata_path) as f:
        return json.load(f)
