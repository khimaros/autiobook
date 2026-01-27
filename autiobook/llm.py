"""llm integration for script and cast generation."""

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar, cast

import litellm

from .config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_THINKING_BUDGET,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = LLM_MAX_RETRIES,
    initial_delay: float = LLM_RETRY_DELAY,
) -> T:
    """retry a function with exponential backoff on API or JSON errors."""
    delay = initial_delay
    last_error: Exception = RuntimeError("unknown error in retry_with_backoff")

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries:
                print(f"  json parse error: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2
        except Exception as e:
            # catch API errors (connection, rate limit, etc.)
            last_error = e
            if attempt < max_retries:
                print(f"  api error: {e}, retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2

    raise last_error


@dataclass
class Character:
    name: str
    description: str  # visual/vocal description for VoiceDesign
    audition_line: str  # short text to generate the reference voice
    aliases: list[str] | None = None  # alternate names for the same character


@dataclass
class ScriptSegment:
    speaker: str
    text: str
    instruction: str  # e.g., "laughing", "whispering", "angry"


def _strip_thinking_tokens(content: str) -> str:
    """remove thinking/reasoning blocks from LLM response."""
    # handle <think>...</think> blocks (DeepSeek, Qwen, etc.)
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    # handle <reasoning>...</reasoning> blocks
    content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
    return content.strip()


def _parse_json_response(content: str) -> dict | list:
    """parse JSON from LLM response, handling markdown code blocks and thinking tokens."""
    content = _strip_thinking_tokens(content)
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()
    try:
        return cast(dict | list, json.loads(content))
    except json.JSONDecodeError:
        # handle trailing garbage (extra data) from some LLMs
        obj, _ = json.JSONDecoder().raw_decode(content)
        return cast(dict | list, obj)


def _query_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    wrapper_keys: List[str] | None = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> dict | list:
    """query LLM via litellm and return JSON data.

    model must include provider prefix (e.g., openai/gpt-4o, anthropic/claude-3-5-sonnet).
    see https://docs.litellm.ai/docs/providers for supported providers.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if thinking_budget > 0:
        # pass via extra_body to bypass litellm's param validation for custom endpoints
        kwargs["extra_body"] = {
            "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
        }

    def _call():
        res = litellm.completion(**kwargs)
        # litellm types are incomplete; access response data via Any
        choices: Any = res.choices
        content: str | None = choices[0].message.content
        if not content:
            raise RuntimeError("llm returned empty content")

        data = _parse_json_response(content)

        # unwrap nested structures when wrapper_keys specified
        if wrapper_keys:
            # handle {"seg": [...]} or {"segments": [...]}
            if isinstance(data, dict):
                for key in wrapper_keys:
                    if key in data:
                        return data[key]
            # handle [{"seg": [...]}] - single-element list wrapping
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                for key in wrapper_keys:
                    if key in data[0]:
                        return data[0][key]
        return data

    return cast(dict | list, retry_with_backoff(_call))


def generate_cast(
    text_sample: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    existing_cast_summary: Optional[str] = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> List[Character]:
    """analyze text to identify characters and generate voice descriptions."""
    context_str = (
        f"\nExisting Cast:\n{existing_cast_summary}\n" if existing_cast_summary else ""
    )

    prompt = f"""Identify book characters. Output ONLY valid JSON, no markdown.

Rules:
1. For each character: n (name), d (vocal description), a (audition line), al (aliases).
2. Vocal description 'd': timbre, pitch, speed, gender, age.
3. Use full name for 'n', variations in 'al'.
4. Omit existing characters unless updating.
{context_str}
Output format (REQUIRED):
{{"c":[{{"n":"Name","d":"voice description","a":"sample line","al":["alias1"]}}]}}
"""

    data = _query_llm_json(
        prompt,
        text_sample,
        model,
        api_base,
        api_key,
        wrapper_keys=["c", "characters"],
        thinking_budget=thinking_budget,
    )

    return [
        Character(
            name=str(c.get("n", c.get("name", ""))),
            description=str(c.get("d", c.get("description", ""))),
            audition_line=str(c.get("a", c.get("audition_line", ""))),
            aliases=c.get("al", c.get("aliases")),
        )
        for c in cast(list, data)
        if isinstance(c, dict)
    ]


def split_text_smart(text: str, max_words: int = 1500) -> List[str]:
    """split text into chunks at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk: List[str] = []
    current_count = 0

    for p in paragraphs:
        word_count = len(p.split())
        if current_count + word_count > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_count = 0

        current_chunk.append(p)
        current_count += word_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def process_script_chunk(
    text_chunk: str,
    characters_list: List[Character],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> List[ScriptSegment]:
    """convert a text chunk into dramatized script segments."""
    cast_str = _format_cast_list(characters_list)

    prompt = f"""Convert text to JSON script. Output ONLY valid JSON, no markdown.

Cast: {cast_str}

Rules:
1. Separate quotes from narration into segments.
2. Narrator speaks all unquoted text.
3. Characters speak only words inside quotes. Use "Extra Female/Male" if unknown.
4. EXACT text only - do not add or remove words.

Output format (REQUIRED - use these exact keys):
{{"seg":[{{"s":"SpeakerName","t":"exact text","i":"mood"}}]}}

Example: "Hi," John said. ->
{{"seg":[{{"s":"John","t":"Hi.","i":"warm"}},{{"s":"Narrator","t":"John said.","i":"narrative"}}]}}
"""

    data = _query_llm_json(
        prompt,
        text_chunk,
        model,
        api_base,
        api_key,
        wrapper_keys=["seg", "segments"],
        thinking_budget=thinking_budget,
    )

    return _parse_script_segments(data)


def _format_cast_list(characters_list: List[Character]) -> str:
    """format cast list for LLM prompts."""
    cast_info = []
    for c in characters_list:
        if c.aliases:
            cast_info.append(f"{c.name} (also known as: {', '.join(c.aliases)})")
        else:
            cast_info.append(c.name)
    return "\n- ".join(cast_info)


def _parse_script_segments(data: list | dict) -> List[ScriptSegment]:
    """parse LLM response into ScriptSegment list with robust error handling."""
    if not isinstance(data, list):
        raise ValueError(f"expected list of segments, got {type(data).__name__}")

    results = []
    for i, s in enumerate(data):
        if not isinstance(s, dict):
            continue
        if "s" not in s or "t" not in s or "i" not in s:
            missing = [k for k in ["s", "t", "i"] if k not in s]
            preview = str(s)[:100] + "..." if len(str(s)) > 100 else str(s)
            raise KeyError(f"segment {i} missing keys {missing}: {preview}")
        results.append(
            ScriptSegment(
                speaker=s["s"],
                text=s["t"],
                instruction=s["i"],
            )
        )
    return results


def fix_missing_segment(
    missing_text: str,
    context_before: str,
    context_after: str,
    characters_list: List[Character],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> List[ScriptSegment]:
    """convert a missing text fragment into script segments using surrounding context."""
    cast_str = _format_cast_list(characters_list)

    prompt = f"""Convert ONLY the "MISSING TEXT" to JSON. Output ONLY valid JSON, no markdown.

Cast: {cast_str}

Rules:
1. Use CONTEXT for speaker/tone but do NOT include context in output.
2. Narrator speaks all unquoted text.
3. Characters speak only words inside quotes.
4. EXACT text only - do not add or remove words.

Output format (REQUIRED - use these exact keys):
{{"seg":[{{"s":"SpeakerName","t":"exact text","i":"mood"}}]}}
"""

    user_content = f"""
CONTEXT BEFORE:
{context_before}

--- MISSING TEXT (convert this) ---
{missing_text}
--- END MISSING TEXT ---

CONTEXT AFTER:
{context_after}
"""

    data = _query_llm_json(
        prompt,
        user_content,
        model,
        api_base,
        api_key,
        wrapper_keys=["seg", "segments"],
        thinking_budget=thinking_budget,
    )

    return _parse_script_segments(data)
