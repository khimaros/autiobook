"""tests for epub extraction functionality."""

from autiobook.epub import extract_text_from_html


class TestExtractTextFromHtml:
    """tests for extract_text_from_html function."""

    def test_simple_paragraph(self):
        """verify basic paragraph extraction."""
        html = b"<p>Hello world.</p>"
        result = extract_text_from_html(html)
        assert result == "Hello world."

    def test_multiple_paragraphs(self):
        """verify multiple paragraphs are separated by blank lines."""
        html = b"<p>First paragraph.</p><p>Second paragraph.</p>"
        result = extract_text_from_html(html)
        assert result == "First paragraph.\n\nSecond paragraph."

    def test_nested_div_with_paragraphs_no_duplication(self):
        """verify div containing p tags extracts leaf nodes as separate paragraphs."""
        html = b"""<div class="bkt1">
<p class="h2a">Survival tips</p>
<p class="hangb">First tip.</p>
<p class="hangb">Second tip.</p>
<p class="hangb">Third tip.</p>
</div>"""
        result = extract_text_from_html(html)
        lines = [line for line in result.split("\n\n") if line.strip()]
        assert len(lines) == 4
        assert lines[0] == "Survival tips"
        assert lines[1] == "First tip."
        assert lines[2] == "Second tip."
        assert lines[3] == "Third tip."

    def test_nested_div_with_bullet_list(self):
        """verify real-world nested structure does not duplicate."""
        html = b"""<div class="bkt1">
<p class="h2a">Midwife Cath's survival tips for a crying baby</p>
<p class="hangb">Have your baby checked to ensure that he is not unwell.</p>
<p class="hangb">Feed your baby.</p>
<p class="hangb">Love your baby.</p>
</div>"""
        result = extract_text_from_html(html)
        # count occurrences of each phrase - should only appear once
        assert result.count("Midwife Cath's survival tips") == 1
        assert result.count("Have your baby checked") == 1
        assert result.count("Feed your baby") == 1
        assert result.count("Love your baby") == 1

    def test_deeply_nested_structure(self):
        """verify deeply nested content tags don't multiply content."""
        html = b"""<div><div><p>Only once.</p></div></div>"""
        result = extract_text_from_html(html)
        assert result.count("Only once.") == 1

    def test_div_with_direct_text_preserved(self):
        """verify div with direct text (no child p) is still extracted."""
        html = b"<div>Direct text in div.</div>"
        result = extract_text_from_html(html)
        assert "Direct text in div." in result

    def test_mixed_nested_and_flat(self):
        """verify mix of nested and flat structures works correctly."""
        html = b"""<div><p>Nested paragraph.</p></div>
<p>Flat paragraph.</p>
<div>Direct div text.</div>"""
        result = extract_text_from_html(html)
        assert result.count("Nested paragraph.") == 1
        assert result.count("Flat paragraph.") == 1
        assert result.count("Direct div text.") == 1

    def test_skip_tags_removed(self):
        """verify script/style tags are removed."""
        html = b"""<p>Visible.</p>
<script>alert('hidden');</script>
<style>.hidden {}</style>
<p>Also visible.</p>"""
        result = extract_text_from_html(html)
        assert "Visible." in result
        assert "Also visible." in result
        assert "alert" not in result
        assert "hidden" not in result

    def test_headings_extracted(self):
        """verify h1-h6 tags are extracted."""
        html = b"<h1>Title</h1><h2>Subtitle</h2><p>Body.</p>"
        result = extract_text_from_html(html)
        assert "Title" in result
        assert "Subtitle" in result
        assert "Body." in result

    def test_whitespace_normalized(self):
        """verify excessive whitespace is collapsed."""
        html = b"<p>Word   with    spaces.</p>"
        result = extract_text_from_html(html)
        assert result == "Word with spaces."

    def test_mixed_content_preservation(self):
        """verify text directly inside a div is preserved even if it contains paragraphs."""
        html = b"""<div class="intro">
    Here is some text directly inside a div.
    <p>Here is a paragraph inside the div.</p>
    Here is more text inside the div <span style="font-weight:bold">with a span</span>.
</div>"""
        result = extract_text_from_html(html)
        assert "Here is some text directly inside a div." in result
        assert "Here is a paragraph inside the div." in result
        assert "Here is more text inside the div with a span." in result
        # Ensure no duplication of the paragraph
        assert result.count("Here is a paragraph inside the div.") == 1
