"""shared test helpers."""

import json
from unittest.mock import MagicMock, patch

import pytest

CAST_REPLY = {
    "characters": [
        {
            "name": "Obligator",
            "description": "d",
            "voice": "v",
            "audition_line": "Model wrote this.",
        }
    ]
}


@pytest.fixture
def cast_prompt():
    """run generate_cast against a stub; return (system prompt sent, characters).

    asserting on the rendered prompt rather than the source keeps the tests
    honest about what the model is actually told, and survives rewrapping.
    """

    def run(**kwargs):
        from autiobook.llm import generate_cast

        payload = json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": json.dumps(CAST_REPLY)},
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode()
        resp = MagicMock()
        resp.read.return_value = payload
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp) as mock_url:
            chars, _ = generate_cast(
                "text", api_base="http://localhost/v1", model="m", **kwargs
            )
        body = json.loads(mock_url.call_args[0][0].data)
        system = next(m for m in body["messages"] if m["role"] == "system")
        return system["content"], chars

    return run
