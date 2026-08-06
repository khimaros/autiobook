"""end-to-end tests for epub3 media overlay export.

builds a real epub, extracts it, fabricates chapter audio plus a timing
manifest, then exports a read-along epub3 and asserts the properties that
silently break playback: unresolved fragment ids, overlapping clips, and
missing package wiring.
"""

import posixpath
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from lxml import etree

from autiobook.config import SAMPLE_RATE
from autiobook.epub import ensure_extracted, load_metadata
from autiobook.overlay import (
    Par,
    anchor_chunks,
    clock,
    merge_pars,
    paragraph_bounds,
    wrap_range,
)

NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "opf": "http://www.idpf.org/2007/opf",
    "smil": "http://www.w3.org/ns/SMIL",
}

# two paragraphs; the first is long enough to clear MIN_CHAPTER_WORDS once
# repeated, and carries inline markup so anchoring is exercised against it.
PARA_ONE = (
    "The harbour lights came on one by one. "
    '"We should go," said Mara, "before the tide turns." '
    "He did not answer her at first."
)
PARA_TWO = (
    "Rain moved across the water in slow grey sheets. "
    "Somewhere behind them a door closed. "
    "The <em>Kestrel</em> rocked against her moorings and was still."
)

# the wrapping div is deliberate: it is a content tag that emits no text of its
# own, so the emitted-paragraph list and the content-tag walk fall out of step.
# real epubs are full of these, and indexing one space with the other puts every
# highlight on the previous paragraph.
CHAPTER_XHTML = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>Chapter One</title></head>
  <body>
    <div class="wrapper">
      <h1>Chapter One</h1>
      <p>{PARA_ONE}</p>
      <p>{PARA_TWO}</p>
      <p>{PARA_ONE}</p>
      <p>{PARA_TWO}</p>
    </div>
  </body>
</html>
"""

CONTAINER = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0"
           xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf"
              media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

NAV = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops">
  <head><title>Contents</title></head>
  <body><nav epub:type="toc"><ol>
    <li><a href="chapter1.xhtml">Chapter One</a></li>
  </ol></nav></body>
</html>
"""

OPF3 = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:test-readalong</dc:identifier>
    <dc:title>Test Book</dc:title>
    <dc:creator>Test Author</dc:creator>
    <dc:language>en</dc:language>
    <meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml"
          properties="nav"/>
    <item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="ch1"/></spine>
</package>
"""

OPF2 = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="uid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="uid">urn:uuid:test-readalong-2</dc:identifier>
    <dc:title>Test Book</dc:title>
    <dc:creator>Test Author</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    <item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine toc="ncx"><itemref idref="ch1"/></spine>
</package>
"""

NCX = """<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="urn:uuid:test-readalong-2"/></head>
  <docTitle><text>Test Book</text></docTitle>
  <navMap>
    <navPoint id="np1" playOrder="1">
      <navLabel><text>Chapter One</text></navLabel>
      <content src="chapter1.xhtml"/>
    </navPoint>
  </navMap>
</ncx>
"""


def build_epub(path: Path, epub2: bool = False) -> Path:
    """write a minimal but valid epub to `path`."""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml", CONTAINER)
        z.writestr("OEBPS/content.opf", OPF2 if epub2 else OPF3)
        z.writestr("OEBPS/chapter1.xhtml", CHAPTER_XHTML)
        if epub2:
            z.writestr("OEBPS/toc.ncx", NCX)
        else:
            z.writestr("OEBPS/nav.xhtml", NAV)
    return path


def fabricate_audio(workdir: Path, meta: dict, chunk_texts: list[str]) -> None:
    """write a silent chapter wav plus a timing manifest covering `chunk_texts`."""
    from autiobook.pooling import timing_manifest_path

    perform = workdir / "perform"
    perform.mkdir(parents=True, exist_ok=True)
    base = meta["chapters"][0]["filename_base"]
    wav_path = perform / f"{base}.wav"

    per_chunk_s = 2.0
    pause_s = 0.5
    total_s = len(chunk_texts) * per_chunk_s + (len(chunk_texts) - 1) * pause_s
    sf.write(
        str(wav_path),
        np.zeros(int(total_s * SAMPLE_RATE), dtype=np.float32),
        SAMPLE_RATE,
    )

    chunks, offset = [], 0.0
    for i, text in enumerate(chunk_texts):
        chunks.append(
            {
                "hash": f"h{i}",
                "start_s": offset,
                "end_s": offset + per_chunk_s,
                "text": text,
                "speaker": "Narrator",
            }
        )
        offset += per_chunk_s + pause_s

    import json

    timing_manifest_path(wav_path).write_text(
        json.dumps({"version": 1, "sample_rate": SAMPLE_RATE, "chunks": chunks})
    )


def chunk_texts_from_source(source: str) -> list[str]:
    """split the extracted text into sentence-ish chunks, as perform would."""
    from autiobook.utils import chunk_text

    out = []
    for para in source.split("\n\n"):
        for sentence in chunk_text(para, 60):
            if sentence.strip():
                out.append(sentence.strip())
    return out


@pytest.fixture
def readalong(tmp_path):
    """build an epub, extract it, fabricate audio, and export the read-along."""

    def _build(epub2: bool = False):
        from autiobook.overlay import export_epub3

        src = build_epub(tmp_path / "book.epub", epub2=epub2)
        workdir = tmp_path / ("wd2" if epub2 else "wd3")
        ensure_extracted(src, workdir)
        meta = load_metadata(workdir)

        source = (
            workdir / "extract" / f"{meta['chapters'][0]['filename_base']}.txt"
        ).read_text()
        fabricate_audio(workdir, meta, chunk_texts_from_source(source))

        out = export_epub3(workdir, tmp_path / "out", bitrate="32k")
        assert out is not None and out.exists()
        return out

    return _build


def read_package(epub_path: Path):
    """return (opf element, opf_dir, zipfile) for a generated epub."""
    zf = zipfile.ZipFile(epub_path)
    container = etree.fromstring(zf.read("META-INF/container.xml"))
    opf_path = container.find(".//container:rootfile", namespaces=NS).get("full-path")
    return etree.fromstring(zf.read(opf_path)), posixpath.dirname(opf_path), zf


class TestClock:
    """SMIL clock value formatting."""

    def test_zero(self):
        assert clock(0) == "0:00:00.000"

    def test_subsecond_precision(self):
        assert clock(1.2345) == "0:00:01.234"

    def test_hours_and_minutes(self):
        assert clock(3723.5) == "1:02:03.500"

    def test_negative_clamped(self):
        assert clock(-5) == "0:00:00.000"


class TestAnchoring:
    """chunk to source-paragraph anchoring."""

    def test_bounds_match_joined_text(self):
        paragraphs = [(0, "alpha beta"), (1, "gamma"), (2, "delta")]
        source = "\n\n".join(t for _, t in paragraphs)
        for (start, end), (_, text) in zip(paragraph_bounds(paragraphs), paragraphs):
            assert source[start:end] == text

    def test_chunks_anchor_in_order(self):
        paragraphs = [(0, "The cat sat down."), (1, "The cat stood up.")]
        source = "\n\n".join(t for _, t in paragraphs)
        chunks = [{"text": "The cat sat down."}, {"text": "The cat stood up."}]
        pieces = anchor_chunks(source, chunks, paragraph_bounds(paragraphs))
        assert [[p[0] for p in c] for c in pieces] == [[0], [1]]

    def test_repeated_text_advances_cursor(self):
        """identical paragraphs must anchor to successive occurrences."""
        paragraphs = [(0, "All the same words here."), (1, "All the same words here.")]
        source = "\n\n".join(t for _, t in paragraphs)
        chunks = [{"text": "All the same words here."}] * 2
        pieces = anchor_chunks(source, chunks, paragraph_bounds(paragraphs))
        assert [[p[0] for p in c] for c in pieces] == [[0], [1]]

    def test_unmatched_chunk_yields_no_anchor(self):
        paragraphs = [(0, "Nothing alike.")]
        source = "Nothing alike."
        chunks = [{"text": "Completely unrelated invented sentence."}]
        assert anchor_chunks(source, chunks, paragraph_bounds(paragraphs)) == [[]]

    def test_chunk_spanning_paragraphs_covers_both(self):
        """one tts chunk often runs across a paragraph break.

        anchoring it only where it starts leaves the later paragraph never
        highlighted while its audio plays, which reads as the highlight
        sticking on the wrong text.
        """
        paragraphs = [
            (0, "Case shrugged. The girl to his right giggled."),
            (1, "The bartender's smile widened. His ugliness was legend."),
        ]
        source = "\n\n".join(t for _, t in paragraphs)
        chunks = [{"text": source}]
        pieces = anchor_chunks(source, chunks, paragraph_bounds(paragraphs))[0]
        assert [p[0] for p in pieces] == [0, 1]
        # the shares are contiguous and cover the whole chunk
        assert pieces[0][1] == 0.0
        assert pieces[-1][2] == 1.0
        assert pieces[0][2] == pieces[1][1]


class TestWrapRange:
    """injecting a span around exactly one chunk's text."""

    def parse(self, xml):
        return etree.fromstring(xml.encode())

    def test_wraps_a_plain_range(self):
        el = self.parse("<p>One two three four.</p>")
        assert wrap_range(el, 4, 13, "x")
        span = el.find("span")
        assert span.get("id") == "x"
        assert span.text == "two three"
        # the paragraph still reads the same
        assert "".join(el.itertext()) == "One two three four."

    def test_wraps_across_collapsed_whitespace(self):
        """extracted offsets are against collapsed text, the source is not."""
        el = self.parse("<p>One   two\n   three four.</p>")
        assert wrap_range(el, 4, 13, "x")
        assert el.find("span").text.split() == ["two", "three"]

    def test_range_inside_a_tail_is_wrapped(self):
        el = self.parse("<p>Start <em>mid</em> after the end.</p>")
        # "after" begins at collapsed offset 10
        assert wrap_range(el, 10, 15, "x")
        span = [e for e in el if e.get("id") == "x"][0]
        assert span.text == "after"
        assert "".join(el.itertext()) == "Start mid after the end."

    def test_range_crossing_markup_is_refused(self):
        """a <par> references one element; splitting an <em> is not worth it."""
        el = self.parse("<p>Start <em>mid</em> after.</p>")
        assert not wrap_range(el, 0, 15, "x")
        assert el.find("span") is None

    def test_out_of_bounds_is_refused(self):
        el = self.parse("<p>Short.</p>")
        assert not wrap_range(el, 0, 999, "x")
        assert not wrap_range(el, 5, 5, "x")


class TestMergePars:
    """chunks sharing an anchor collapse into one <par>."""

    def test_chunks_in_same_paragraph_merge(self):
        chunks = [
            {"start_s": 0.0, "end_s": 1.0},
            {"start_s": 1.5, "end_s": 2.5},
            {"start_s": 3.0, "end_s": 4.0},
        ]
        pars = merge_pars(
            chunks,
            [[("a", 0.0, 1.0)], [("a", 0.0, 1.0)], [("b", 0.0, 1.0)]],
        )
        assert [p.fragment_id for p in pars] == ["a", "b"]
        assert pars[0].clip_begin == 0.0
        assert pars[1].clip_end == 4.0

    def test_straddling_chunk_becomes_two_pars(self):
        """the reported symptom: a chunk crossing a break must light both.

        the second paragraph's audio played while the first stayed
        highlighted for the whole chunk.
        """
        chunks = [{"start_s": 10.0, "end_s": 20.0}]
        pars = merge_pars(chunks, [[("a", 0.0, 0.4), ("b", 0.4, 1.0)]])
        assert [p.fragment_id for p in pars] == ["a", "b"]
        assert pars[0].clip_begin == 10.0
        assert pars[0].clip_end == pars[1].clip_begin == 14.0
        assert pars[1].clip_end == 20.0

    def test_clips_are_gapless(self):
        """each par runs up to the next so inter-chunk pauses still play."""
        chunks = [
            {"start_s": 0.0, "end_s": 1.0},
            {"start_s": 1.5, "end_s": 2.5},
        ]
        pars = merge_pars(chunks, [[("a", 0.0, 1.0)], [("b", 0.0, 1.0)]])
        assert pars[0].clip_end == pars[1].clip_begin == 1.5

    def test_unanchored_chunks_are_dropped(self):
        chunks = [{"start_s": 0.0, "end_s": 1.0}, {"start_s": 2.0, "end_s": 3.0}]
        pars = merge_pars(chunks, [[], [("a", 0.0, 1.0)]])
        assert [p.fragment_id for p in pars] == ["a"]

    def test_no_anchors_yields_nothing(self):
        assert merge_pars([{"start_s": 0.0, "end_s": 1.0}], [[]]) == []

    def test_par_is_comparable(self):
        assert Par("a", 0.0, 1.0) == Par("a", 0.0, 1.0)


class TestReadAlongExport:
    """end-to-end structure of the generated epub3."""

    def test_package_is_epub3(self, readalong):
        opf, _, _ = read_package(readalong())
        assert opf.get("version") == "3.0"

    def test_document_links_its_overlay(self, readalong):
        opf, _, _ = read_package(readalong())
        items = opf.find("opf:manifest", namespaces=NS).findall(
            "opf:item", namespaces=NS
        )
        overlaid = [i for i in items if i.get("media-overlay")]
        assert len(overlaid) == 1
        smil_id = overlaid[0].get("media-overlay")
        smil = next(i for i in items if i.get("id") == smil_id)
        assert smil.get("media-type") == "application/smil+xml"

    def test_audio_is_embedded(self, readalong):
        opf, opf_dir, zf = read_package(readalong())
        audio = [
            i
            for i in opf.find("opf:manifest", namespaces=NS).findall(
                "opf:item", namespaces=NS
            )
            if i.get("media-type") == "audio/mpeg"
        ]
        assert len(audio) == 1
        assert zf.read(posixpath.join(opf_dir, audio[0].get("href")))

    def test_every_fragment_resolves(self, readalong):
        """the failure that silently breaks highlighting in every reader."""
        opf, opf_dir, zf = read_package(readalong())
        items = opf.find("opf:manifest", namespaces=NS).findall(
            "opf:item", namespaces=NS
        )
        by_id = {i.get("id"): i for i in items}
        doc_item = next(i for i in items if i.get("media-overlay"))
        smil_href = by_id[doc_item.get("media-overlay")].get("href")

        smil = etree.fromstring(zf.read(posixpath.join(opf_dir, smil_href)))
        doc = etree.fromstring(zf.read(posixpath.join(opf_dir, doc_item.get("href"))))
        doc_ids = {e.get("id") for e in doc.iter() if e.get("id")}

        pars = smil.findall(".//smil:par", namespaces=NS)
        assert pars
        for par in pars:
            src = par.find("smil:text", namespaces=NS).get("src")
            assert "#" in src
            assert src.split("#", 1)[1] in doc_ids

    def test_fragments_point_at_the_text_being_spoken(self, readalong):
        """the highlight must land on the paragraph the audio is reading.

        anchoring works in emitted-paragraph space while the document walk is
        in content-tag space; conflating them shifts every fragment onto an
        earlier element and the highlight trails the narration.
        """
        opf, opf_dir, zf = read_package(readalong())
        items = opf.find("opf:manifest", namespaces=NS).findall(
            "opf:item", namespaces=NS
        )
        by_id = {i.get("id"): i for i in items}
        doc_item = next(i for i in items if i.get("media-overlay"))
        smil = etree.fromstring(
            zf.read(
                posixpath.join(
                    opf_dir, by_id[doc_item.get("media-overlay")].get("href")
                )
            )
        )
        doc = etree.fromstring(zf.read(posixpath.join(opf_dir, doc_item.get("href"))))
        by_frag = {e.get("id"): e for e in doc.iter() if e.get("id")}

        frags = [
            p.find("smil:text", namespaces=NS).get("src").split("#", 1)[1]
            for p in smil.findall(".//smil:par", namespaces=NS)
        ]
        targets = [by_frag[f] for f in frags]

        # a paragraph fragment may contain injected chunk spans, but must never
        # contain another paragraph fragment -- that is what the off-by-one
        # produced, with the wrapper div swallowing every paragraph under it.
        blocks = {id(e) for e in targets if e.tag.rsplit("}", 1)[-1] != "span"}
        for el in targets:
            nested = [d for d in el.iterdescendants() if id(d) in blocks]
            text = " ".join("".join(el.itertext()).split())
            assert not nested, f"fragment {text[:50]!r} contains other paragraphs"

        first = " ".join("".join(targets[0].itertext()).split())
        assert first == "Chapter One", f"first fragment points at {first[:60]!r}"

    def test_clips_are_monotonic(self, readalong):
        opf, opf_dir, zf = read_package(readalong())
        items = opf.find("opf:manifest", namespaces=NS).findall(
            "opf:item", namespaces=NS
        )
        by_id = {i.get("id"): i for i in items}
        doc_item = next(i for i in items if i.get("media-overlay"))
        smil = etree.fromstring(
            zf.read(
                posixpath.join(
                    opf_dir, by_id[doc_item.get("media-overlay")].get("href")
                )
            )
        )

        def to_s(v):
            h, m, s = v.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        prev = 0.0
        for par in smil.findall(".//smil:par", namespaces=NS):
            audio = par.find("smil:audio", namespaces=NS)
            begin, end = to_s(audio.get("clipBegin")), to_s(audio.get("clipEnd"))
            assert end > begin
            assert begin >= prev - 1e-6
            prev = end

    def test_durations_are_declared(self, readalong):
        opf, _, _ = read_package(readalong())
        metas = opf.find("opf:metadata", namespaces=NS).findall(
            "opf:meta", namespaces=NS
        )
        durations = [m for m in metas if m.get("property") == "media:duration"]
        assert any(m.get("refines") for m in durations), "per-overlay duration missing"
        assert any(not m.get("refines") for m in durations), "total duration missing"
        assert any(m.get("property") == "media:active-class" for m in metas)

    def test_original_documents_are_preserved(self, readalong):
        """rebuilding must not drop the head or its stylesheets."""
        opf, opf_dir, zf = read_package(readalong())
        doc_item = next(
            i
            for i in opf.find("opf:manifest", namespaces=NS).findall(
                "opf:item", namespaces=NS
            )
            if i.get("media-overlay")
        )
        doc = etree.fromstring(zf.read(posixpath.join(opf_dir, doc_item.get("href"))))
        names = {e.tag.rsplit("}", 1)[-1] for e in doc.iter()}
        assert "head" in names and "title" in names
        assert "em" in names, "inline markup was lost"

    def test_mimetype_is_first_and_stored(self, readalong):
        with zipfile.ZipFile(readalong()) as z:
            first = z.infolist()[0]
            assert first.filename == "mimetype"
            assert first.compress_type == zipfile.ZIP_STORED

    def test_epub2_source_is_upgraded_with_nav(self, readalong):
        """media overlays require epub3, which requires a nav document."""
        opf, _, _ = read_package(readalong(epub2=True))
        assert opf.get("version") == "3.0"
        nav = [
            i
            for i in opf.find("opf:manifest", namespaces=NS).findall(
                "opf:item", namespaces=NS
            )
            if "nav" in (i.get("properties") or "")
        ]
        assert len(nav) == 1

    def test_epub2_upgrade_still_anchors(self, readalong):
        opf, opf_dir, zf = read_package(readalong(epub2=True))
        items = opf.find("opf:manifest", namespaces=NS).findall(
            "opf:item", namespaces=NS
        )
        by_id = {i.get("id"): i for i in items}
        doc_item = next(i for i in items if i.get("media-overlay"))
        smil = etree.fromstring(
            zf.read(
                posixpath.join(
                    opf_dir, by_id[doc_item.get("media-overlay")].get("href")
                )
            )
        )
        assert smil.findall(".//smil:par", namespaces=NS)
