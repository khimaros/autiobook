"""epub3 media overlay (read-along) export.

rebuilds the source epub with a SMIL overlay per narrated chapter so readers
highlight text in sync with the narration. overlay granularity follows the
audio chunk boundaries already recorded in the timing manifest; where several
chunks land in the same source element they merge into a single <par>, since a
<par> can only reference one text fragment.

the original epub is copied entry-for-entry and only the narrated content
documents, the package document, and the new smil/audio files are touched.
"""

import io
import json
import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from lxml import etree  # type: ignore

from .config import (
    CONTENT_TAGS,
    EPUB_EXT,
    MP3_EXT,
    NAV_FILE,
    OVERLAY_ACTIVE_CLASS,
    OVERLAY_DIR,
    OVERLAY_HIGHLIGHT_CSS,
    OVERLAY_ID_PREFIX,
    SKIP_TAGS,
    SMIL_EXT,
)
from .epub import paragraph_bounds

CONTAINER_PATH = "META-INF/container.xml"
NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "xhtml": "http://www.w3.org/1999/xhtml",
    "smil": "http://www.w3.org/ns/SMIL",
    "epub": "http://www.idpf.org/2007/ops",
    "ncx": "http://www.daisy.org/z3986/2005/ncx/",
}
NCX_MEDIA_TYPE = "application/x-dtbncx+xml"
SMIL_MEDIA_TYPE = "application/smil+xml"
MP3_MEDIA_TYPE = "audio/mpeg"
XHTML_MEDIA_TYPE = "application/xhtml+xml"


def _local(tag: Any) -> str:
    """local name of an lxml element tag, namespace stripped."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def clock(seconds: float) -> str:
    """format seconds as a SMIL clock value (H:MM:SS.mmm)."""
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    s = (total_ms // 1000) % 60
    m = (total_ms // 60000) % 60
    h = total_ms // 3600000
    return f"{h}:{m:02d}:{s:02d}.{ms:03d}"


# --- source element enumeration -------------------------------------------


def _loose_text(el: etree._Element) -> str:
    """text of an element excluding nested content-tag subtrees."""
    parts = [el.text or ""]
    for child in el:
        if isinstance(child.tag, str) and _local(child.tag) not in CONTENT_TAGS:
            parts.append(_loose_text(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _paragraph_text(el: etree._Element) -> str:
    """text of a content element, mirroring extract_paragraphs_from_html.

    a container holding nested content tags anywhere beneath it contributes
    only its own loose text, so nested paragraphs are not counted twice.
    """
    if any(
        isinstance(d.tag, str) and _local(d.tag) in CONTENT_TAGS
        for d in el.iterdescendants()
    ):
        return " ".join(_loose_text(el).split())
    return " ".join("".join(el.itertext()).split())


def content_elements(root: etree._Element) -> list[etree._Element]:
    """content elements in document order, excluding any under a skip tag.

    yields the same sequence -- and therefore the same indices -- as
    extract_paragraphs_from_html's decomposed walk, but over a tree that can be
    serialized back out with the document's head intact.
    """
    out = []
    for el in root.iter():
        if not isinstance(el.tag, str) or _local(el.tag) not in CONTENT_TAGS:
            continue
        if any(_local(p.tag) in SKIP_TAGS for p in el.iterancestors()):
            continue
        out.append(el)
    return out


# --- span injection --------------------------------------------------------


def _qname(el: etree._Element, name: str) -> str:
    """`name` in the same namespace as `el`, so injected markup stays xhtml."""
    tag = el.tag
    if isinstance(tag, str) and tag.startswith("{"):
        return f"{{{tag[1 : tag.index('}')]}}}{name}"
    return name


def _text_slots(el: etree._Element) -> list[tuple[etree._Element, str, str]]:
    """(holder, attr, raw text) for every text node under el, in document order."""
    slots: list[tuple[etree._Element, str, str]] = []
    if el.text:
        slots.append((el, "text", el.text))
    for child in el:
        if isinstance(child.tag, str):
            slots.extend(_text_slots(child))
        if child.tail:
            slots.append((child, "tail", child.tail))
    return slots


def _collapsed_positions(
    slots: list[tuple[etree._Element, str, str]],
) -> tuple[str, list[tuple[int, int]]]:
    """collapse slot text the way extraction does, keeping a position per char.

    each collapsed character maps back to the (slot, raw offset) it came from,
    which is what lets a span be wrapped around an offset range computed
    against the extracted text.
    """
    chars: list[str] = []
    origin: list[tuple[int, int]] = []
    pending = False
    for slot_idx, (_, _, raw) in enumerate(slots):
        for raw_idx, ch in enumerate(raw):
            if ch.isspace():
                if chars:
                    pending = True
                continue
            if pending:
                chars.append(" ")
                origin.append((slot_idx, raw_idx))
                pending = False
            chars.append(ch)
            origin.append((slot_idx, raw_idx))
    return "".join(chars), origin


def wrap_range(
    el: etree._Element, start: int, end: int, span_id: str, expected: str | None = None
) -> bool:
    """wrap [start, end) of el's collapsed text in a span carrying `span_id`.

    returns False when the range crosses inline markup, since a <par> can only
    reference one element and splitting an <em> in two to satisfy that is not
    worth the fidelity risk; the caller falls back to the whole paragraph.

    `expected` is the paragraph text the offsets were computed against. it is
    checked before any edit: a container element walks differently here than
    during extraction (which skips nested content tags), and wrapping against a
    mismatched string lands the span mid-word.
    """
    slots = _text_slots(el)
    collapsed, origin = _collapsed_positions(slots)
    if expected is not None and collapsed != expected:
        return False
    if not 0 <= start < end <= len(collapsed):
        return False

    start_slot, start_raw = origin[start]
    end_slot, end_raw = origin[end - 1]
    if start_slot != end_slot:
        return False

    holder, attr, raw = slots[start_slot]
    before, inner, after = (
        raw[:start_raw],
        raw[start_raw : end_raw + 1],
        raw[end_raw + 1 :],
    )
    if not inner.strip():
        return False

    span = etree.Element(_qname(el, "span"))
    span.set("id", span_id)
    span.text = inner
    span.tail = after
    if attr == "text":
        holder.text = before
        holder.insert(0, span)
    else:
        parent = holder.getparent()
        if parent is None:
            return False
        holder.tail = before
        parent.insert(parent.index(holder) + 1, span)
    return True


# --- anchoring -------------------------------------------------------------


@dataclass
class Par:
    """one <par>: a text fragment paired with a clip of the chapter audio."""

    fragment_id: str
    clip_begin: float
    clip_end: float


Piece = tuple[int, float, float]  # (paragraph index, start share, end share)
Placement = tuple[str, float, float]  # (fragment id, start share, end share)


def _paragraph_pieces(
    bounds: list[tuple[int, int]], start: int, end: int
) -> list[Piece]:
    """paragraphs overlapping [start, end) and each one's share of the span.

    a chunk that runs across a paragraph break is split proportionally by how
    much of its text falls in each paragraph. speech rate is near enough to
    constant that character count approximates where the boundary lands in
    time, which beats leaving the later paragraph unhighlighted entirely.
    """
    overlaps: list[tuple[int, int]] = []
    total = 0
    for i, (p_start, p_end) in enumerate(bounds):
        if p_end <= start:
            continue
        if p_start >= end:
            break
        width = min(end, p_end) - max(start, p_start)
        if width > 0:
            overlaps.append((i, width))
            total += width
    if not overlaps or total <= 0:
        return []

    pieces: list[Piece] = []
    seen = 0
    for i, width in overlaps:
        begin = seen / total
        seen += width
        pieces.append((i, begin, seen / total))
    return pieces


def locate_chunks(
    source: str, chunks: list[dict[str, Any]]
) -> list[tuple[int, int] | None]:
    """char span of each chunk within the chapter text, or None.

    chunks are matched in order with a forward cursor, so a repeated phrase
    resolves to its next occurrence rather than the first.
    """
    from .dramatize import _find_text_in_source

    cursor = 0
    out: list[tuple[int, int] | None] = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            out.append(None)
            continue
        span = _find_text_in_source(text, source, cursor)
        if not span:
            out.append(None)
            continue
        cursor = span[1]
        out.append(span)
    return out


def anchor_chunks(
    source: str, chunks: list[dict[str, Any]], bounds: list[tuple[int, int]]
) -> list[list[Piece]]:
    """map each chunk to the paragraphs it covers, with each one's share."""
    return [
        _paragraph_pieces(bounds, *span) if span else []
        for span in locate_chunks(source, chunks)
    ]


def merge_pars(
    chunks: list[dict[str, Any]], placements: list[list[Placement]]
) -> list[Par]:
    """build one <par> per fragment, merging consecutive chunks that share one.

    each placement is (fragment id, start share, end share) of its chunk. a
    chunk wrapped in its own span is one placement spanning the whole chunk, so
    its clip times are the recorded ones; a chunk that fell back to paragraph
    anchoring is apportioned across the paragraphs it covers.

    clip ranges are extended to meet the following par so the pauses between
    chunks are played rather than skipped, and audio from chunks that placed
    nowhere still plays under the preceding fragment.
    """
    pars: list[Par] = []
    for chunk, places in zip(chunks, placements):
        begin = float(chunk.get("start_s", 0.0))
        end = float(chunk.get("end_s", begin))
        span = max(0.0, end - begin)
        for fragment_id, from_share, to_share in places:
            piece_begin = begin + span * from_share
            piece_end = begin + span * to_share
            if pars and pars[-1].fragment_id == fragment_id:
                pars[-1].clip_end = piece_end
                continue
            pars.append(Par(fragment_id, piece_begin, piece_end))

    for i in range(len(pars) - 1):
        pars[i].clip_end = pars[i + 1].clip_begin
    return pars


# --- smil ------------------------------------------------------------------


def build_smil(
    smil_href: str, doc_href: str, audio_href: str, pars: list[Par]
) -> bytes:
    """render a media overlay document for one chapter."""
    base = posixpath.dirname(smil_href)
    text_rel = posixpath.relpath(doc_href, base) if base else doc_href
    audio_rel = posixpath.relpath(audio_href, base) if base else audio_href

    root = etree.Element(
        "{%s}smil" % NS["smil"],
        nsmap={None: NS["smil"], "epub": NS["epub"]},
        version="3.0",
    )
    body = etree.SubElement(root, "{%s}body" % NS["smil"])
    seq = etree.SubElement(body, "{%s}seq" % NS["smil"])
    seq.set("id", "seq1")
    seq.set("{%s}textref" % NS["epub"], text_rel)

    for i, par in enumerate(pars, start=1):
        par_el = etree.SubElement(seq, "{%s}par" % NS["smil"])
        par_el.set("id", f"par{i}")
        text_el = etree.SubElement(par_el, "{%s}text" % NS["smil"])
        text_el.set("src", f"{text_rel}#{par.fragment_id}")
        audio_el = etree.SubElement(par_el, "{%s}audio" % NS["smil"])
        audio_el.set("src", audio_rel)
        audio_el.set("clipBegin", clock(par.clip_begin))
        audio_el.set("clipEnd", clock(par.clip_end))

    return cast(
        bytes,
        etree.tostring(root, xml_declaration=True, encoding="utf-8", pretty_print=True),
    )


# --- content document rewriting -------------------------------------------


def place_chunks(
    doc: etree._ElementTree,
    paragraphs: list[tuple[int, str]],
    bounds: list[tuple[int, int]],
    spans: list[tuple[int, int] | None],
) -> tuple[list[list[Placement]], int]:
    """decide which element each chunk highlights, preferring its own span.

    a chunk that sits inside one paragraph gets a span wrapped around exactly
    its text, so the <par> can use the chunk's recorded clip times. anything
    else -- crossing a paragraph break, or a range that crosses inline markup
    -- falls back to the containing paragraphs, apportioned by character count.

    returns the placements and how many chunks got an exact span.
    """
    elements = content_elements(doc.getroot())
    # paragraphs are indexed by position in the emitted list; the document walk
    # is indexed by content-tag position, and the two diverge as soon as a
    # content tag emits no text of its own.
    tag_index = [tag_idx for tag_idx, _ in paragraphs]
    used = {el.get("id") for el in doc.getroot().iter() if el.get("id")}
    para_ids: dict[int, str] = {}
    placements: list[list[Placement]] = []
    exact = 0

    def paragraph_id(para: int) -> str | None:
        """id of the paragraph element, assigning one if it has none."""
        if para in para_ids:
            return para_ids[para]
        if not 0 <= para < len(tag_index):
            return None
        idx = tag_index[para]
        if idx >= len(elements):
            return None
        el = elements[idx]
        existing = str(el.get("id") or "")
        if not existing:
            existing = _unique_id(f"{OVERLAY_ID_PREFIX}{idx}", used)
            el.set("id", existing)
        used.add(existing)
        para_ids[para] = existing
        return existing

    for n, span in enumerate(spans):
        if span is None:
            placements.append([])
            continue
        start, end = span
        pieces = _paragraph_pieces(bounds, start, end)
        if not pieces:
            placements.append([])
            continue

        if len(pieces) == 1:
            para = pieces[0][0]
            idx = tag_index[para] if 0 <= para < len(tag_index) else -1
            p_start = bounds[para][0]
            if 0 <= idx < len(elements):
                span_id = _unique_id(f"{OVERLAY_ID_PREFIX}c{n}", used)
                if wrap_range(
                    elements[idx],
                    start - p_start,
                    end - p_start,
                    span_id,
                    expected=paragraphs[para][1],
                ):
                    used.add(span_id)
                    placements.append([(span_id, 0.0, 1.0)])
                    exact += 1
                    continue

        placed = [
            (frag, from_share, to_share)
            for para, from_share, to_share in pieces
            if (frag := paragraph_id(para)) is not None
        ]
        placements.append(placed)

    return placements, exact


def _unique_id(candidate: str, used: set[str]) -> str:
    """`candidate`, suffixed until it does not collide."""
    if candidate not in used:
        return candidate
    n = 1
    while f"{candidate}_{n}" in used:
        n += 1
    return f"{candidate}_{n}"


def inject_highlight_css(doc: etree._ElementTree) -> None:
    """add a style rule for the active-fragment class used during playback."""
    root = doc.getroot()
    head = next((e for e in root.iter() if _local(e.tag) == "head"), None)
    if head is None:
        return
    ns = root.tag.rsplit("}", 1)[0][1:] if "}" in str(root.tag) else None
    style = etree.SubElement(head, "{%s}style" % ns if ns else "style")
    style.set("type", "text/css")
    style.text = OVERLAY_HIGHLIGHT_CSS


# --- package document ------------------------------------------------------


def opf_path(zf: zipfile.ZipFile) -> str:
    """locate the package document via META-INF/container.xml."""
    root = etree.fromstring(zf.read(CONTAINER_PATH))
    rootfile = root.find(".//container:rootfile", namespaces=NS)
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("epub container.xml has no rootfile")
    return str(rootfile.get("full-path"))


def _manifest(opf: etree._Element) -> etree._Element:
    el = opf.find("opf:manifest", namespaces=NS)
    if el is None:
        raise ValueError("package document has no manifest")
    return el


def _metadata(opf: etree._Element) -> etree._Element:
    el = opf.find("opf:metadata", namespaces=NS)
    if el is None:
        raise ValueError("package document has no metadata")
    return el


def manifest_id_by_href(opf: etree._Element) -> dict[str, str]:
    """map manifest href -> item id."""
    out = {}
    for item in _manifest(opf).findall("opf:item", namespaces=NS):
        href, iid = item.get("href"), item.get("id")
        if href and iid:
            out[href] = iid
    return out


def add_manifest_item(
    opf: etree._Element, iid: str, href: str, media_type: str, **attrs: str
) -> None:
    item = etree.SubElement(_manifest(opf), "{%s}item" % NS["opf"])
    item.set("id", iid)
    item.set("href", href)
    item.set("media-type", media_type)
    for k, v in attrs.items():
        item.set(k.replace("_", "-"), v)


def add_meta(opf: etree._Element, prop: str, value: str, refines: str = "") -> None:
    """add an epub3 <meta property=...> to the package metadata."""
    meta = etree.SubElement(_metadata(opf), "{%s}meta" % NS["opf"])
    meta.set("property", prop)
    if refines:
        meta.set("refines", refines)
    meta.text = value


def build_nav(opf: etree._Element, zf: zipfile.ZipFile, opf_dir: str) -> bytes | None:
    """derive an epub3 nav document from the ncx, for epub2 sources.

    epub3 requires a nav document; without one an upgraded package fails
    validation even though readers may tolerate it.
    """
    ncx_href = next(
        (
            item.get("href")
            for item in _manifest(opf).findall("opf:item", namespaces=NS)
            if item.get("media-type") == NCX_MEDIA_TYPE and item.get("href")
        ),
        None,
    )
    entries: list[tuple[str, str]] = []
    if ncx_href:
        ncx_path = posixpath.normpath(posixpath.join(opf_dir, ncx_href))
        try:
            ncx = etree.fromstring(zf.read(ncx_path))
        except (KeyError, etree.XMLSyntaxError):
            ncx = None
        if ncx is not None:
            for point in ncx.iter():
                if _local(point.tag) != "navPoint":
                    continue
                label = next((t for t in point.iter() if _local(t.tag) == "text"), None)
                content = next(
                    (c for c in point.iter() if _local(c.tag) == "content"), None
                )
                if label is None or content is None or not content.get("src"):
                    continue
                src = str(content.get("src"))
                # nav lives beside the opf; ncx hrefs are relative to the ncx.
                src = posixpath.normpath(
                    posixpath.join(posixpath.dirname(ncx_href), src)
                )
                entries.append((src, (label.text or "").strip() or src))

    if not entries:
        return None

    items = "\n".join(
        f'      <li><a href="{href}">{_escape(title)}</a></li>'
        for href, title in entries
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" '
        'xmlns:epub="http://www.idpf.org/2007/ops">\n'
        "  <head><title>Contents</title></head>\n"
        "  <body>\n"
        '    <nav epub:type="toc" id="toc">\n'
        "      <ol>\n"
        f"{items}\n"
        "      </ol>\n"
        "    </nav>\n"
        "  </body>\n"
        "</html>\n"
    ).encode("utf-8")


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --- packaging -------------------------------------------------------------


def verify_alignment(root: etree._Element, paragraphs: list[tuple[int, str]]) -> bool:
    """confirm the lxml walk indexes the same elements the extractor saw.

    the two parsers agree for well-formed xhtml, but a document that only
    survives html-style error recovery can diverge; anchoring against a
    mismatched index would highlight the wrong text.
    """
    elements = content_elements(root)
    for idx, text in paragraphs:
        if idx >= len(elements) or _paragraph_text(elements[idx]) != text:
            return False
    return True


def encode_mp3(wav_path: Path, bitrate: str) -> bytes:
    """encode a chapter wav to mp3 bytes for embedding."""
    import io

    from pydub import AudioSegment  # type: ignore

    buf = io.BytesIO()
    AudioSegment.from_wav(str(wav_path)).export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()


def _read_doc(zf: zipfile.ZipFile, path: str) -> bytes | None:
    """read a content document, tolerating percent-encoded manifest hrefs."""
    from urllib.parse import quote, unquote

    for candidate in (path, unquote(path), quote(path)):
        try:
            return zf.read(candidate)
        except KeyError:
            continue
    return None


@dataclass
class ChapterOverlay:
    """everything generated for one narrated chapter."""

    doc_zip_path: str
    doc_href: str
    doc_bytes: bytes
    smil_href: str
    smil_bytes: bytes
    audio_href: str
    audio_bytes: bytes
    duration_s: float
    pars: int
    exact: int  # chunks highlighted by their own span rather than a paragraph


def build_chapter_overlay(
    zf: zipfile.ZipFile,
    opf_dir: str,
    doc_href: str,
    source_text: str,
    chunks: list[dict[str, Any]],
    wav_path: Path,
    stem: str,
    bitrate: str,
) -> ChapterOverlay | None:
    """anchor one chapter's chunks and render its overlay, or None if unusable."""
    from .epub import extract_paragraphs_from_html

    doc_zip_path = posixpath.normpath(posixpath.join(opf_dir, doc_href))
    raw = _read_doc(zf, doc_zip_path)
    if raw is None:
        print(f"epub3: {doc_href}: not found in epub, skipping overlay")
        return None

    try:
        doc = etree.parse(
            io.BytesIO(raw),
            etree.XMLParser(resolve_entities=False, strip_cdata=False),
        )
    except etree.XMLSyntaxError as e:
        print(f"epub3: {doc_href}: not well-formed xml ({e}), skipping overlay")
        return None

    paragraphs = extract_paragraphs_from_html(raw)
    if not verify_alignment(doc.getroot(), paragraphs):
        print(f"epub3: {doc_href}: element walk diverged, skipping overlay")
        return None

    # chunk offsets are resolved against the extracted text but applied to the
    # document. if extract/ predates a change in the extractor the two describe
    # different strings, and every span lands somewhere arbitrary.
    if source_text.rstrip("\n") != "\n\n".join(t for _, t in paragraphs).rstrip("\n"):
        print(
            f"epub3: {doc_href}: extract/ is stale for this chapter "
            "(re-run `extract --force`), skipping overlay"
        )
        return None

    bounds = paragraph_bounds(paragraphs)
    spans = locate_chunks(source_text, chunks)
    placements, exact = place_chunks(doc, paragraphs, bounds, spans)
    if not any(placements):
        print(f"epub3: {doc_href}: no chunks anchored, skipping overlay")
        return None

    inject_highlight_css(doc)
    pars = merge_pars(chunks, placements)
    if not pars:
        return None

    from .export import get_wav_duration_ms

    duration_s = get_wav_duration_ms(wav_path) / 1000.0
    pars[-1].clip_end = max(pars[-1].clip_end, duration_s)

    smil_href = f"{OVERLAY_DIR}/{stem}{SMIL_EXT}"
    audio_href = f"{OVERLAY_DIR}/{stem}{MP3_EXT}"

    return ChapterOverlay(
        doc_zip_path=doc_zip_path,
        doc_href=doc_href,
        doc_bytes=etree.tostring(doc, xml_declaration=True, encoding="utf-8"),
        smil_href=smil_href,
        smil_bytes=build_smil(smil_href, doc_href, audio_href, pars),
        audio_href=audio_href,
        audio_bytes=encode_mp3(wav_path, bitrate),
        duration_s=duration_s,
        pars=len(pars),
        exact=exact,
    )


def _wire_package(
    opf: etree._Element, overlays: list[ChapterOverlay], nav_added: bool
) -> None:
    """add manifest entries, media-overlay links and media:duration metadata."""
    href_to_id = manifest_id_by_href(opf)
    total = 0.0

    for i, ov in enumerate(overlays, start=1):
        smil_id = f"{OVERLAY_ID_PREFIX}-smil-{i}"
        audio_id = f"{OVERLAY_ID_PREFIX}-audio-{i}"
        add_manifest_item(opf, audio_id, ov.audio_href, MP3_MEDIA_TYPE)
        add_manifest_item(opf, smil_id, ov.smil_href, SMIL_MEDIA_TYPE)

        doc_id = href_to_id.get(ov.doc_href)
        for item in _manifest(opf).findall("opf:item", namespaces=NS):
            if item.get("id") == doc_id:
                item.set("media-overlay", smil_id)
                break

        add_meta(opf, "media:duration", clock(ov.duration_s), refines=f"#{smil_id}")
        total += ov.duration_s

    add_meta(opf, "media:duration", clock(total))
    add_meta(opf, "media:active-class", OVERLAY_ACTIVE_CLASS)

    if nav_added:
        add_manifest_item(
            opf,
            f"{OVERLAY_ID_PREFIX}-nav",
            NAV_FILE,
            XHTML_MEDIA_TYPE,
            properties="nav",
        )
    opf.set("version", "3.0")


def export_epub3(
    workdir: Path,
    output_dir: Path,
    bitrate: str = "64k",
    epub_path: Path | None = None,
    chapters: list[int] | None = None,
) -> Path | None:
    """rebuild the source epub with media overlays for every narrated chapter."""
    from .config import WAV_EXT
    from .epub import load_metadata, source_epub_path
    from .export import _book_slug
    from .pooling import timing_manifest_path
    from .resume import get_command_dir, list_chapters

    meta = load_metadata(workdir)
    src = epub_path or source_epub_path(workdir)
    if not src or not src.exists():
        print(
            "epub3: source epub not found; pass --epub to point at the original "
            f"({src if src else 'no path recorded at extract time'})"
        )
        return None

    # workdirs extracted before hrefs were recorded still work: the chapter
    # selection is deterministic, so hrefs can be recovered from the epub.
    if any(not c.get("href") for c in meta["chapters"]):
        from .epub import parse_epub

        by_index = {c.index: c.href for c in parse_epub(src)[0].chapters}
        for c in meta["chapters"]:
            if not c.get("href"):
                c["href"] = by_index.get(c["index"], "")

    source_dir = next(
        (
            d
            for d in [workdir / "perform", workdir / "synthesize"]
            if d.exists() and list_chapters(meta, d, output_dir, source_ext=WAV_EXT)
        ),
        None,
    )
    if not source_dir:
        print("epub3: no wav files found")
        return None

    chapter_paths = list_chapters(
        meta, source_dir, output_dir, chapters_filter=chapters, source_ext=WAV_EXT
    )
    info_map = {c["index"]: c for c in meta["chapters"]}
    extract_dir = get_command_dir(workdir, "extract")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{_book_slug(meta['title'])}{EPUB_EXT}"

    with zipfile.ZipFile(src) as zf:
        opf_zip_path = opf_path(zf)
        opf_dir = posixpath.dirname(opf_zip_path)
        opf = etree.fromstring(zf.read(opf_zip_path))
        was_epub2 = str(opf.get("version", "")).startswith("2")

        overlays: list[ChapterOverlay] = []
        for idx, wav_p, _ in sorted(chapter_paths, key=lambda x: x[0]):
            info = info_map.get(idx)
            if not info or not info.get("href"):
                continue
            tpath = timing_manifest_path(wav_p)
            txt_path = extract_dir / f"{info['filename_base']}.txt"
            if not tpath.exists() or not txt_path.exists():
                continue
            try:
                chunks = json.loads(tpath.read_text()).get("chunks", [])
            except (OSError, ValueError):
                continue
            if not chunks:
                continue

            ov = build_chapter_overlay(
                zf,
                opf_dir,
                info["href"],
                txt_path.read_text(encoding="utf-8"),
                chunks,
                wav_p,
                info["filename_base"],
                bitrate,
            )
            if ov:
                print(
                    f"epub3: {info['href']}: {ov.pars} par(s), "
                    f"{ov.exact}/{len(chunks)} chunk-exact"
                )
                overlays.append(ov)

        if not overlays:
            print("epub3: no chapters could be anchored")
            return None

        nav_bytes = build_nav(opf, zf, opf_dir) if was_epub2 else None
        _wire_package(opf, overlays, nav_added=nav_bytes is not None)

        replacements = {ov.doc_zip_path: ov.doc_bytes for ov in overlays}
        replacements[opf_zip_path] = etree.tostring(
            opf, xml_declaration=True, encoding="utf-8"
        )
        additions: dict[str, bytes] = {}
        for ov in overlays:
            additions[posixpath.join(opf_dir, ov.smil_href)] = ov.smil_bytes
            additions[posixpath.join(opf_dir, ov.audio_href)] = ov.audio_bytes
        if nav_bytes:
            additions[posixpath.join(opf_dir, NAV_FILE)] = nav_bytes

        _write_epub(zf, out_path, replacements, additions)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"epub3: wrote {out_path} ({len(overlays)} chapter overlay(s), "
        f"{size_mb:.1f} MiB)"
    )
    if was_epub2:
        print("epub3: source was epub2; upgraded package to 3.0")
    return out_path


def _write_epub(
    src: zipfile.ZipFile,
    out_path: Path,
    replacements: dict[str, bytes],
    additions: dict[str, bytes],
) -> None:
    """copy the source epub entry-for-entry, applying replacements/additions."""
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as out:
        # mimetype must be the first entry and stored uncompressed
        out.writestr(
            zipfile.ZipInfo("mimetype"),
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        for info in src.infolist():
            if info.filename == "mimetype" or info.filename in additions:
                continue
            out.writestr(
                info.filename, replacements.get(info.filename, src.read(info.filename))
            )
        for name, data in additions.items():
            out.writestr(name, data)
