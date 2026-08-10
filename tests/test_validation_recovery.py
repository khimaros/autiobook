"""tests for recovery when a chunk validates down to an empty script.

regression cover for the "no segments provided" fix loop: a fully hallucinated
chunk was emptied by hallucination removal, and the empty-script guard then fed
its own placeholder string back to the llm as missing text forever.
"""

import json
from unittest.mock import patch

import pytest

from autiobook.llm import ScriptSegment

SOURCE = (
    "“This scam of yours, when it’s over, you erase this goddam thing.”\n\n"
    "CASE DIDN’T UNDERSTAND the Zionites.\n"
)

# the shapes the old code leaked into llm prompts as if they were book text
PLACEHOLDERS = ("no segments provided", "no script found")


class TestEmptyScriptReportsSourceText:
    """an empty script is entirely missing, so the source text is the fragment."""

    def test_validate_chunk_empty_segments(self):
        from autiobook.dramatize import validate_chunk

        result = validate_chunk(SOURCE, [])

        assert len(result.missing) == 1
        fragment = result.missing[0][0]
        assert "erase this goddam thing" in fragment
        assert not any(p in fragment for p in PLACEHOLDERS)

    def test_validate_script_missing_file(self, tmp_path):
        from autiobook.dramatize import validate_script

        txt_path = tmp_path / "11_EIGHT.txt"
        txt_path.write_text(SOURCE, encoding="utf-8")

        result = validate_script(txt_path, tmp_path / "absent.json")

        assert len(result.missing) == 1
        fragment = result.missing[0][0]
        assert "erase this goddam thing" in fragment
        assert not any(p in fragment for p in PLACEHOLDERS)

    def test_validate_script_empty_segment_list(self, tmp_path):
        from autiobook.dramatize import validate_script

        txt_path = tmp_path / "11_EIGHT.txt"
        txt_path.write_text(SOURCE, encoding="utf-8")
        script_path = tmp_path / "11_EIGHT.json"
        script_path.write_text(json.dumps({"version": 1, "segments": []}))

        result = validate_script(txt_path, script_path)

        assert "erase this goddam thing" in result.missing[0][0]

    def test_empty_source_and_empty_script_is_clean(self):
        """nothing to say and nothing said: no fragment to hand the llm."""
        from autiobook.dramatize import validate_chunk

        assert validate_chunk("   \n", []).missing == []


class TestFullyHallucinatedChunkRecovers:
    """the neuromancer 11_EIGHT failure: one hallucinated segment, no recovery.

    the model returned a chat preamble that the json repair turn wrapped in the
    segment schema. removing it emptied the script, and every later attempt was
    spent converting a placeholder back and forth."""

    def test_redo_replaces_the_emptied_chunk(self):
        from autiobook.dramatize import process_script_chunk_with_validation

        preamble = ScriptSegment(
            speaker="Narrator",
            text="Thank you for sharing this text. It appears to be an excerpt.",
            instruction="excited",
        )
        good = ScriptSegment("Dix", SOURCE.strip(), "neutral")

        seeds = []

        def fake_convert(*args, **kwargs):
            seeds.append(kwargs.get("seed"))
            return [preamble] if len(seeds) == 1 else [good]

        with patch(
            "autiobook.dramatize.process_script_chunk", side_effect=fake_convert
        ):
            with patch("autiobook.dramatize.fix_missing_segment") as mock_fix:
                segments = process_script_chunk_with_validation(
                    SOURCE, [], model="model"
                )

        assert [s.text for s in segments] == [SOURCE.strip()]
        # a whole-chunk gap never reaches the prompt written for filling gaps
        assert mock_fix.call_count == 0
        # first call is the initial conversion, then a reseeded redo
        assert seeds[0] is None
        assert seeds[1] is not None

    def test_redo_seed_moves_with_the_attempt(self):
        """an identical seeded request would replay the reply that just failed."""
        from autiobook.config import active_seed
        from autiobook.dramatize import process_script_chunk_with_validation

        junk = [ScriptSegment("Narrator", "Entirely invented sentence.", "neutral")]
        seeds = []

        def fake_convert(*args, **kwargs):
            seeds.append(kwargs.get("seed"))
            return list(junk)

        with patch(
            "autiobook.dramatize.process_script_chunk", side_effect=fake_convert
        ):
            with pytest.raises(Exception):
                process_script_chunk_with_validation(SOURCE, [], model="model")

        redos = [s for s in seeds if s is not None]
        assert len(redos) == len(set(redos))
        assert all(s > active_seed() for s in redos)


class TestEmptyResponseIsRecoverable:
    """a reply with no content must not kill the run mid-chapter."""

    def _response(self, content, reasoning=None):
        msg = {"role": "assistant", "content": content}
        if reasoning is not None:
            msg["reasoning_content"] = reasoning
        return json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}]})

    def _urlopen(self, bodies):
        from unittest.mock import MagicMock

        def side_effect(*args, **kwargs):
            resp = MagicMock()
            resp.read.return_value = bodies.pop(0).encode()
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        return side_effect

    def test_empty_content_becomes_feedback(self):
        from autiobook.llm import Character, process_script_chunk

        good = '{"seg":[{"s":"John","t":"Hello.","i":"happy"}]}'
        bodies = [self._response("", "thinking but no answer"), self._response(good)]

        with patch(
            "urllib.request.urlopen", side_effect=self._urlopen(bodies)
        ) as mock_url:
            segments = process_script_chunk(
                "Hello.",
                [Character("John", "male voice", "Hello.", None)],
                model="m",
                api_base="http://localhost/v1",
            )

        assert len(segments) == 1
        # exactly one retry: no pointless backoff resend of an identical body
        assert mock_url.call_count == 2
        body = json.loads(mock_url.call_args_list[1][0][0].data)
        assert "no content" in body["messages"][-1]["content"].lower()

    def test_answer_left_in_reasoning_is_salvaged(self):
        from autiobook.llm import Character, process_script_chunk

        good = '{"seg":[{"s":"John","t":"Hello.","i":"happy"}]}'
        bodies = [self._response("", good)]

        with patch(
            "urllib.request.urlopen", side_effect=self._urlopen(bodies)
        ) as mock_url:
            segments = process_script_chunk(
                "Hello.",
                [Character("John", "male voice", "Hello.", None)],
                model="m",
                api_base="http://localhost/v1",
            )

        assert len(segments) == 1
        assert mock_url.call_count == 1

    def test_empty_response_is_not_retried_by_backoff(self):
        from autiobook.llm import EmptyResponseError, retry_with_backoff

        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise EmptyResponseError("empty")

        with pytest.raises(EmptyResponseError):
            retry_with_backoff(fn)

        assert calls["n"] == 1


class TestJsonRepairFeedback:
    """the repair turn must ask for the original task, not a reformat."""

    def test_parse_error_feedback_restates_task(self):
        from autiobook.llm import _feedback_for_error

        err = json.JSONDecodeError("Expecting value", "Thank you for sharing", 0)
        feedback = _feedback_for_error(
            "Thank you for sharing this text.", err, expected_shape='{"segments": []}'
        )

        lowered = feedback.lower()
        assert "do not" in lowered
        assert "original" in lowered or "redo" in lowered


@pytest.mark.parametrize("placeholder", PLACEHOLDERS)
def test_placeholder_strings_are_gone(placeholder):
    """the sentinels must not survive anywhere in the validation path."""
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "autiobook" / "dramatize.py"
    body = src.read_text(encoding="utf-8")
    # allowed only in comments explaining the removed behaviour
    code_lines = [
        line
        for line in body.splitlines()
        if placeholder in line and not line.lstrip().startswith("#")
    ]
    assert code_lines == []
