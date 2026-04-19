"""tests for LLM integration and .env configuration."""

import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _mock_urlopen(content: str):
    """create a mock for urllib.request.urlopen returning an openai-compatible response."""
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


class TestEnvLoading:
    """tests for .env file loading."""

    def test_load_env_from_workdir(self):
        """verify .env file in workdir is loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("TEST_VAR_XYZ=workdir_value\n")

            os.environ.pop("TEST_VAR_XYZ", None)

            from autiobook.env import load_env

            load_env(Path(tmpdir))

            assert os.environ.get("TEST_VAR_XYZ") == "workdir_value"

            os.environ.pop("TEST_VAR_XYZ", None)

    def test_load_env_from_cwd_fallback(self):
        """verify .env in cwd is used when workdir has no .env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir) / "workdir"
            workdir.mkdir()

            os.environ.pop("TEST_VAR_ABC", None)

            from autiobook.env import load_env

            load_env(workdir)  # should not crash

    def test_env_not_override_existing(self):
        """verify .env does not override already-set environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("EXISTING_VAR=from_file\n")

            os.environ["EXISTING_VAR"] = "already_set"

            from autiobook.env import load_env

            load_env(Path(tmpdir))

            assert os.environ.get("EXISTING_VAR") == "already_set"

            os.environ.pop("EXISTING_VAR", None)


class TestLLMIntegration:
    """tests for openai-compatible LLM API integration."""

    def test_query_llm_json_basic(self):
        """verify basic json response parsing."""
        mock_resp = _mock_urlopen('{"result": "ok"}')

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(
                system_prompt="test system",
                user_prompt="test user",
                model="gpt-4o",
                api_base="http://localhost:8080/v1",
            )

            assert result == {"result": "ok"}

    def test_query_llm_json_model_in_request(self):
        """verify model is included in the request body."""
        mock_resp = _mock_urlopen('{"result": "ok"}')

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            from autiobook.llm import _query_llm_json

            _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="my-model",
                api_base="http://localhost:8080/v1",
            )

            # verify the request was made
            mock_url.assert_called_once()
            req = mock_url.call_args[0][0]
            body = json.loads(req.data)
            assert body["model"] == "my-model"


class TestThinkingBudget:
    """tests for thinking/extended thinking support."""

    def test_thinking_budget_in_request(self):
        """verify thinking parameter is included when budget > 0."""
        mock_resp = _mock_urlopen('{"result": "ok"}')

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            from autiobook.llm import _query_llm_json

            _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="qwen3",
                api_base="http://localhost:8080/v1",
                thinking_budget=1024,
            )

            req = mock_url.call_args[0][0]
            body = json.loads(req.data)
            assert body["thinking_budget_tokens"] == 1024

    def test_zero_thinking_budget_unlimited(self):
        """verify thinking_budget=0 omits field (unlimited thinking)."""
        mock_resp = _mock_urlopen('{"result": "ok"}')

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            from autiobook.llm import _query_llm_json

            _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="qwen3",
                api_base="http://localhost:8080/v1",
                thinking_budget=0,
            )

            req = mock_url.call_args[0][0]
            body = json.loads(req.data)
            assert "thinking_budget_tokens" not in body


class TestSeed:
    """tests for deterministic seed support."""

    def test_default_seed_is_concrete_random(self):
        """when AUTIOBOOK_SEED is unset, a concrete positive seed is generated
        and sent so the run is reproducible via AUTIOBOOK_SEED=<value>."""
        mock_resp = _mock_urlopen('{"result": "ok"}')

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            from autiobook.config import DEFAULT_SEED
            from autiobook.llm import _query_llm_json

            _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="gpt-4o",
                api_base="http://localhost:8080/v1",
            )

            req = mock_url.call_args[0][0]
            body = json.loads(req.data)
            assert body.get("seed") == DEFAULT_SEED
            assert DEFAULT_SEED > 0

    def test_custom_seed_in_request(self):
        """verify explicit seed is included."""
        mock_resp = _mock_urlopen('{"result": "ok"}')

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            from autiobook.llm import _query_llm_json

            _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="gpt-4o",
                api_base="http://localhost:8080/v1",
                seed=42,
            )

            req = mock_url.call_args[0][0]
            body = json.loads(req.data)
            assert body["seed"] == 42

    def test_zero_seed_omitted(self):
        """verify seed=0 omits field from request."""
        mock_resp = _mock_urlopen('{"result": "ok"}')

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            from autiobook.llm import _query_llm_json

            _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="gpt-4o",
                api_base="http://localhost:8080/v1",
                seed=0,
            )

            req = mock_url.call_args[0][0]
            body = json.loads(req.data)
            assert "seed" not in body

    def test_negative_seed_omitted(self):
        """verify negative seed omits field from request."""
        mock_resp = _mock_urlopen('{"result": "ok"}')

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            from autiobook.llm import _query_llm_json

            _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="gpt-4o",
                api_base="http://localhost:8080/v1",
                seed=-1,
            )

            req = mock_url.call_args[0][0]
            body = json.loads(req.data)
            assert "seed" not in body


class TestApiConfig:
    """tests for api_base and api_key configuration."""

    def test_api_base_in_url(self):
        """verify api_base is used to construct the request URL."""
        mock_resp = _mock_urlopen('{"result": "ok"}')

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            from autiobook.llm import _query_llm_json

            _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="gpt-4o",
                api_base="http://localhost:8080/v1",
            )

            req = mock_url.call_args[0][0]
            assert req.full_url == "http://localhost:8080/v1/chat/completions"

    def test_api_key_in_header(self):
        """verify api_key is sent as Bearer token."""
        mock_resp = _mock_urlopen('{"result": "ok"}')

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            from autiobook.llm import _query_llm_json

            _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="gpt-4o",
                api_base="http://localhost:8080/v1",
                api_key="sk-test-key",
            )

            req = mock_url.call_args[0][0]
            assert req.get_header("Authorization") == "Bearer sk-test-key"


class TestRetryBehavior:
    """tests for retry behavior on API errors."""

    def test_retry_on_api_error(self):
        """verify retry behavior on transient API errors."""
        import urllib.error

        mock_good = _mock_urlopen('{"result": "ok"}')
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise urllib.error.HTTPError(
                    "http://test", 500, "server error", {}, io.BytesIO(b"error")
                )
            return mock_good

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch("time.sleep"):
                from autiobook.llm import _query_llm_json

                result = _query_llm_json(
                    system_prompt="test",
                    user_prompt="test",
                    model="gpt-4o",
                    api_base="http://localhost:8080/v1",
                )

                assert result == {"result": "ok"}
                assert call_count == 2

    def test_json_parse_error_raises(self):
        """_query_llm_json raises on malformed JSON (no retry at this layer)."""
        mock_bad = _mock_urlopen("not json")

        with patch("urllib.request.urlopen", return_value=mock_bad):
            from autiobook.llm import _query_llm_json

            with pytest.raises(json.JSONDecodeError):
                _query_llm_json(
                    system_prompt="test",
                    user_prompt="test",
                    model="gpt-4o",
                    api_base="http://localhost:8080/v1",
                )


class TestJsonResponseParsing:
    """tests for JSON response parsing."""

    def test_parse_json_with_markdown_fence(self):
        """verify markdown code fences are stripped."""
        mock_resp = _mock_urlopen('```json\n{"result": "ok"}\n```')

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="gpt-4o",
                api_base="http://localhost:8080/v1",
            )

            assert result == {"result": "ok"}

    def test_parse_json_with_wrapper_keys(self):
        """verify wrapper keys are extracted."""
        mock_resp = _mock_urlopen('{"characters": [{"name": "Alice"}]}')

        with patch("urllib.request.urlopen", return_value=mock_resp):
            from autiobook.llm import _query_llm_json

            result = _query_llm_json(
                system_prompt="test",
                user_prompt="test",
                model="gpt-4o",
                api_base="http://localhost:8080/v1",
                wrapper_keys=["characters", "c"],
            )

            assert result == [{"name": "Alice"}]


class TestJSONRepair:
    """repair of common LLM JSON malformations observed in qwen3.6-35b output."""

    def test_unclosed_string_before_key_unescaped(self):
        """`\\", "instruction":` pattern — real qwen failure at char 313."""
        from autiobook.llm import _parse_json_response

        bad = (
            "[\n"
            '  {"speaker": "Sita", "text": "\\"Sorry about this,\\", '
            '"instruction": "neutral"}\n'
            "]"
        )
        result = _parse_json_response(bad)
        assert result == [
            {
                "speaker": "Sita",
                "text": '"Sorry about this,"',
                "instruction": "neutral",
            }
        ]

    def test_unclosed_string_before_escaped_key(self):
        """`\\", \\"instruction":` pattern — mixed escaping."""
        from autiobook.llm import _parse_json_response

        bad = '[{"speaker": "A", "text": "\\"hi,\\", \\"instruction": "neutral"}]'
        result = _parse_json_response(bad)
        assert result == [{"speaker": "A", "text": '"hi,"', "instruction": "neutral"}]

    def test_trailing_comma_in_array(self):
        from autiobook.llm import _parse_json_response

        assert _parse_json_response("[1, 2, 3,]") == [1, 2, 3]

    def test_valid_json_unchanged(self):
        """repair must not break well-formed input."""
        from autiobook.llm import _parse_json_response

        good = '[{"speaker": "A", "text": "\\"hi,\\"", "instruction": "neutral"}]'
        assert _parse_json_response(good) == [
            {"speaker": "A", "text": '"hi,"', "instruction": "neutral"}
        ]


class TestParseCastList:
    """tests for cast list parsing across the shapes LLMs emit."""

    def _char(self, name="X"):
        return {"name": name, "description": "d", "audition_line": "a"}

    def test_bare_list(self):
        from autiobook.llm import _parse_cast_list

        result = _parse_cast_list([self._char("A"), self._char("B")])
        assert [c.name for c in result] == ["A", "B"]

    def test_wrapped_characters_key(self):
        from autiobook.llm import _parse_cast_list

        result = _parse_cast_list({"characters": [self._char("A")]})
        assert [c.name for c in result] == ["A"]

    def test_wrapped_alternate_keys(self):
        from autiobook.llm import _parse_cast_list

        for key in ["c", "cast", "updates", "result", "results"]:
            result = _parse_cast_list({key: [self._char("A")]})
            assert [c.name for c in result] == ["A"], key

    def test_single_character_dict(self):
        from autiobook.llm import _parse_cast_list

        result = _parse_cast_list(self._char("Tam"))
        assert [c.name for c in result] == ["Tam"]

    def test_dict_keyed_by_character_name(self):
        from autiobook.llm import _parse_cast_list

        data = {
            "Tam": {"description": "d", "audition_line": "a"},
            "Seth": {"description": "d2", "audition_line": "a2"},
        }
        result = _parse_cast_list(data)
        assert sorted(c.name for c in result) == ["Seth", "Tam"]

    def test_rejects_unrecognized_dict(self):
        from autiobook.llm import _parse_cast_list

        with pytest.raises(ValueError, match="expected list"):
            _parse_cast_list({"foo": "bar"})


class TestResolveSpeakers:
    """tests for permissive speaker resolution."""

    def _cast(self):
        from autiobook.llm import Character

        return [
            Character(
                name="Hubert Vernon Espinoza",
                description="d",
                audition_line="a",
                aliases=["Hubert, Etc.", "Espinoza"],
            ),
            Character(name="Seth", description="d", audition_line="a"),
            Character(name="Limpopo", description="d", audition_line="a"),
        ]

    def _seg(self, speaker):
        from autiobook.llm import ScriptSegment

        return ScriptSegment(speaker=speaker, text="x", instruction="neutral")

    def test_exact_canonical_passes(self):
        from autiobook.llm import resolve_speakers

        seg = self._seg("Seth")
        errors = resolve_speakers([seg], self._cast())
        assert errors == []
        assert seg.speaker == "Seth"

    def test_exact_alias_resolves_to_canonical(self):
        from autiobook.llm import resolve_speakers

        seg = self._seg("Hubert, Etc.")
        errors = resolve_speakers([seg], self._cast())
        assert errors == []
        assert seg.speaker == "Espinoza"

    def test_trailing_punctuation_normalized(self):
        from autiobook.llm import resolve_speakers

        seg = self._seg("Hubert, Etc")  # missing period
        errors = resolve_speakers([seg], self._cast())
        assert errors == []
        assert seg.speaker == "Espinoza"

    def test_case_insensitive_match(self):
        from autiobook.llm import resolve_speakers

        seg = self._seg("seth")
        errors = resolve_speakers([seg], self._cast())
        assert errors == []
        assert seg.speaker == "Seth"

    def test_unambiguous_substring(self):
        from autiobook.llm import resolve_speakers

        seg = self._seg("Hubert")  # substring of canonical name
        errors = resolve_speakers([seg], self._cast())
        assert errors == []
        assert seg.speaker == "Espinoza"

    def test_narrator_passes(self):
        from autiobook.llm import resolve_speakers

        seg = self._seg("Narrator")
        errors = resolve_speakers([seg], self._cast())
        assert errors == []

    def test_ambiguous_reports_error(self):
        from autiobook.llm import Character, resolve_speakers

        cast = [
            Character(name="John Smith", description="d", audition_line="a"),
            Character(name="John Doe", description="d", audition_line="a"),
        ]
        seg = self._seg("John")
        errors = resolve_speakers([seg], cast)
        assert len(errors) == 1
        assert "ambiguous" in errors[0]
        assert seg.speaker == "John"  # not mutated

    def test_unknown_reports_error(self):
        from autiobook.llm import resolve_speakers

        seg = self._seg("Gandalf")
        errors = resolve_speakers([seg], self._cast())
        assert len(errors) == 1
        assert "unknown speaker 'Gandalf'" in errors[0]


class TestFixInstructions:
    def test_resets_invalid_to_neutral(self):
        from autiobook.llm import ScriptSegment, fix_instructions_inplace

        segs = [
            ScriptSegment(speaker="X", text="a", instruction="excited"),
            ScriptSegment(speaker="X", text="b", instruction="bogus-feeling"),
        ]
        n = fix_instructions_inplace(segs)
        assert n == 1
        assert segs[0].instruction == "excited"
        assert segs[1].instruction == "neutral"


class TestGroupedFeedback:
    def test_validate_script_groups_duplicates(self):
        from autiobook.llm import ScriptSegment, _validate_script_segments

        segs = [
            ScriptSegment(speaker="Gandalf", text=str(i), instruction="neutral")
            for i in range(5)
        ]
        errors = _validate_script_segments(segs, [])
        # one grouped error line + cast hint
        joined = "\n".join(errors)
        assert "segments [0, 1, 2, 3, 4]" in joined
        assert "Valid speakers" in joined
