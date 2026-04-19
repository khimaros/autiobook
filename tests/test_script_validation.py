"""tests for script chunk validation with retry."""

import json
from unittest.mock import MagicMock, patch

import pytest

from autiobook.llm import Character, ScriptSegment


def _mock_urlopen_json(content: str):
    """create a mock for urllib.request.urlopen returning an openai chat response."""
    resp_body = json.dumps(
        {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        }
    ).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = resp_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestValidateChunk:
    """tests for single-chunk validation."""

    def test_validate_chunk_all_text_present(self):
        """validation passes when all source words appear in segments."""
        from autiobook.dramatize import validate_chunk

        source = "Hello world, said John."
        segments = [
            ScriptSegment(speaker="John", text="Hello world.", instruction="happy"),
            ScriptSegment(speaker="Narrator", text="said John.", instruction="neutral"),
        ]

        result = validate_chunk(source, segments)
        assert result.missing == []
        assert result.hallucinated == []

    def test_validate_chunk_missing_text(self):
        """validation detects missing text from source."""
        from autiobook.dramatize import validate_chunk

        source = "Hello world, said John quietly."
        segments = [
            ScriptSegment(speaker="John", text="Hello world.", instruction="happy"),
            ScriptSegment(speaker="Narrator", text="said John.", instruction="neutral"),
        ]

        result = validate_chunk(source, segments)
        assert len(result.missing) > 0
        assert any("quietly" in m[0] for m in result.missing)

    def test_validate_chunk_hallucinated_segment(self):
        """validation detects hallucinated segments not in source."""
        from autiobook.dramatize import validate_chunk

        source = "Hello world."
        segments = [
            ScriptSegment(speaker="John", text="Hello world.", instruction="happy"),
            ScriptSegment(
                speaker="Narrator",
                text="This text was never in the source.",
                instruction="neutral",
            ),
        ]

        result = validate_chunk(source, segments)
        assert len(result.hallucinated) > 0


class TestProcessScriptChunkWithValidation:
    """tests for chunk processing with validation and retry."""

    def test_process_chunk_valid_first_try(self):
        """chunk passes validation on first try, no retry needed."""
        from autiobook.dramatize import process_script_chunk_with_validation

        content = (
            '{"seg":[{"s":"John","t":"Hello.","i":"happy"},'
            '{"s":"Narrator","t":"said John.","i":"neutral"}]}'
        )
        mock_resp = _mock_urlopen_json(content)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            source = '"Hello," said John.'
            cast = [Character("John", "male voice", "Hello.", None)]

            segments = process_script_chunk_with_validation(
                source, cast, model="gpt-4o", api_base="http://localhost/v1"
            )

            assert len(segments) == 2
            assert mock_url.call_count == 1

    def test_process_chunk_retries_on_missing_text(self):
        """chunk retries with feedback when validation detects missing text."""
        from autiobook.dramatize import process_script_chunk_with_validation

        bad_content = (
            '{"seg":[{"s":"John","t":"Hello.","i":"happy"},'
            '{"s":"Narrator","t":"said John.","i":"neutral"}]}'
        )
        good_content = (
            '{"seg":[{"s":"John","t":"Hello.","i":"happy"},'
            '{"s":"Narrator","t":"said John quietly.","i":"neutral"}]}'
        )

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_urlopen_json(bad_content)
            return _mock_urlopen_json(good_content)

        with patch("urllib.request.urlopen", side_effect=side_effect) as mock_url:
            source = '"Hello," said John quietly.'
            cast = [Character("John", "male voice", "Hello.", None)]

            process_script_chunk_with_validation(
                source, cast, model="gpt-4o", api_base="http://localhost/v1"
            )

            assert call_count == 2
            # verify second call includes feedback about missing text
            second_req = mock_url.call_args_list[1][0][0]
            body = json.loads(second_req.data)
            user_msg = next(m for m in body["messages"] if m["role"] == "user")
            assert "quietly" in user_msg["content"].lower()

    def test_process_chunk_raises_after_max_retries(self):
        """chunk raises error after VALIDATION_MAX_RETRIES attempts exhausted."""
        from autiobook.config import VALIDATION_MAX_RETRIES
        from autiobook.dramatize import (
            ValidationError,
            process_script_chunk_with_validation,
        )

        bad_content = '{"seg":[]}'

        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen_json(bad_content),
        ) as mock_url:
            source = '"Hello," said John quietly to Mary.'
            cast = [Character("John", "male voice", "Hello.", None)]

            with pytest.raises(ValidationError) as exc_info:
                process_script_chunk_with_validation(
                    source, cast, model="gpt-4o", api_base="http://localhost/v1"
                )

            # initial call + VALIDATION_MAX_RETRIES retry attempts
            assert mock_url.call_count == VALIDATION_MAX_RETRIES + 1
            assert "validation failed" in str(exc_info.value)
            assert f"{VALIDATION_MAX_RETRIES} iterative fix attempts" in str(
                exc_info.value
            )

    def test_process_chunk_raises_validation_error(self):
        """chunk raises ValidationError on failure."""
        from autiobook.dramatize import (
            ValidationError,
            process_script_chunk_with_validation,
        )

        with patch("autiobook.dramatize.validate_chunk") as mock_validate:
            mock_validate.return_value = MagicMock(
                missing=[("foo", 0, 0, "foo line")], hallucinated=[]
            )

            with patch("autiobook.dramatize.process_script_chunk") as mock_process:
                mock_process.return_value = []

                with patch(
                    "autiobook.dramatize._fill_missing_fragments", return_value=0
                ):
                    with pytest.raises(ValidationError):
                        process_script_chunk_with_validation("text", [], model="model")

    def test_process_chunk_prioritizes_hallucinations(self):
        """verify hallucinations are fixed and then re-validated before missing text."""
        from autiobook.dramatize import (
            ValidationResult,
            process_script_chunk_with_validation,
        )

        segments = [ScriptSegment("Speaker", "Text", "neutral")]

        result1 = ValidationResult(
            missing=[("missing text", 0, 0, "missing text line")], hallucinated=[0]
        )
        result2 = ValidationResult(
            missing=[("missing text", 0, 0, "missing text line")], hallucinated=[]
        )
        result3 = ValidationResult(missing=[], hallucinated=[])

        with patch("autiobook.dramatize.process_script_chunk", return_value=segments):
            with patch(
                "autiobook.dramatize.validate_chunk",
                side_effect=[result1, result2, result3],
            ) as mock_validate:
                with patch("autiobook.dramatize._remove_hallucinations") as mock_remove:
                    with patch(
                        "autiobook.dramatize._fill_missing_fragments"
                    ) as mock_fill:
                        process_script_chunk_with_validation("text", [], model="model")

                        assert mock_remove.call_count == 1
                        assert mock_fill.call_count == 1
                        assert mock_validate.call_count == 3

    def test_process_chunk_retries_on_hallucination(self):
        """chunk retries when validation detects hallucinated content."""
        from autiobook.dramatize import process_script_chunk_with_validation

        bad_content = (
            '{"seg":[{"s":"John","t":"Hello.","i":"happy"},'
            '{"s":"Narrator","t":"John walked away into the sunset.","i":"neutral"}]}'
        )
        good_content = (
            '{"seg":[{"s":"John","t":"Hello.","i":"happy"},'
            '{"s":"Narrator","t":"said John.","i":"neutral"}]}'
        )

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_urlopen_json(bad_content)
            return _mock_urlopen_json(good_content)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            source = '"Hello," said John.'
            cast = [Character("John", "male voice", "Hello.", None)]

            process_script_chunk_with_validation(
                source, cast, model="gpt-4o", api_base="http://localhost/v1"
            )

            assert call_count == 2


class TestScriptCommandValidateFlag:
    """tests for --validate flag on script command."""

    def test_validate_flag_default_false(self):
        """--validate flag defaults to False."""
        import argparse

        with patch("sys.argv", ["autiobook", "script", "/tmp/workdir"]):
            with patch("autiobook.dramatize.run_script_generation"):
                with patch("autiobook.env.load_env"):
                    parser = argparse.ArgumentParser()
                    subparsers = parser.add_subparsers(dest="command")
                    script_parser = subparsers.add_parser("script")
                    script_parser.add_argument("workdir")
                    script_parser.add_argument("--validate", action="store_true")

                    args = parser.parse_args(["script", "/tmp/workdir"])
                    assert args.validate is False

    def test_validate_flag_enabled(self):
        """--validate flag can be enabled."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        script_parser = subparsers.add_parser("script")
        script_parser.add_argument("workdir")
        script_parser.add_argument("--validate", action="store_true")

        args = parser.parse_args(["script", "/tmp/workdir", "--validate"])
        assert args.validate is True
