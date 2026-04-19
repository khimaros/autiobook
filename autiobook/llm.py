"""llm integration for script and cast generation."""

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar, cast

from .config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_SEED,
    DEFAULT_THINKING_BUDGET,
    EMOTION_KEYS,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
    LLM_TIMEOUT,
    RETAINED_SPEAKERS,
    VALIDATION_MAX_RETRIES,
)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = LLM_MAX_RETRIES,
    initial_delay: float = LLM_RETRY_DELAY,
) -> T:
    """retry a function with exponential backoff on API errors."""
    delay = initial_delay
    last_error: Exception = RuntimeError("unknown error in retry_with_backoff")

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
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


# shared prompt components for script generation
SCRIPT_RULES_COMMON = """Script Generation Rules:

- Each segment MUST correspond EXACTLY to text from the source. Do not add or omit text.
- Split the source text into segments, IDEALLY each segment 2-3 sentences in length.
- ALWAYS split character quotes from narration and unspoken text.
- Use "Narrator" as the speaker for ALL unquoted text, including attribution like "John said,"
- Use "Retained" as the speaker for text which shouldn't be spoken: section markers, \
chapter numbers, roman numerals, formatting artifacts, etc. Include the EXACT text.
- Use other characters from the [Character List] for SPOKEN TEXT ONLY. Fallback to \
"Extra Female" or "Extra Male" if the character is unclear.
- Use speaker names EXACTLY as listed FIRST in the [Character List] (the short \
form). Match punctuation and capitalization. Do NOT use the longer variants \
shown after "also:"; those are for recognition only."""

_EMOTION_LIST = ", ".join(EMOTION_KEYS)

SCRIPT_OUTPUT_FORMAT = f"""Script Segment Format (JSON):

Output a single JSON object with key "segments" whose value is a list of \
segments. Each segment has this shape:

```
{{"speaker":"Speaker Name", "text":"exact source text", "instruction":"<emotion>"}}
```

Valid instruction values: {_EMOTION_LIST}
Always use one of these values. Use "neutral" as the default.
"""

# realistic example showing attribution before quote, quote, attribution after quote
SCRIPT_EXAMPLE = """Example Source Input:

```
[iv]

Mary said, "Let's go."

He shook his head. "No," John said quietly.
```

Output (JSON object with "segments" list):

```
{"segments": [
  {"speaker": "Retained", "text": "[iv]"},
  {"speaker": "Narrator", "text": "Mary said,"},
  {"speaker": "Mary", "text": "Let's go.", "instruction": "excited"},
  {"speaker": "Narrator", "text": "He shook his head."},
  {"speaker": "John", "text": "No,", "instruction": "sad"},
  {"speaker": "Narrator", "text": "John said quietly."}
]}
```
"""

SCRIPT_EXPECTED_SHAPE = (
    '{"segments": [{"speaker": ..., "text": ..., "instruction": ...}, ...]}'
)

SCRIPT_GENERATION_COMMON = f"""
{SCRIPT_OUTPUT_FORMAT}

{SCRIPT_EXAMPLE}

{SCRIPT_RULES_COMMON}
"""


_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|reasoning)>(.*?)</(?:think|reasoning)>", flags=re.DOTALL
)


def _strip_thinking_tokens(content: str) -> str:
    """remove thinking/reasoning blocks from LLM response."""
    return _THINKING_BLOCK_RE.sub("", content).strip()


def _extract_inline_reasoning(content: str) -> str:
    """extract concatenated text from any <think>/<reasoning> blocks in content."""
    blocks = _THINKING_BLOCK_RE.findall(content)
    return "\n\n".join(b.strip() for b in blocks if b.strip())


# common keys that follow a value in script/cast segments; used for JSON repair.
_SEGMENT_KEYS = (
    r"speaker|text|instruction|name|description|audition_line|aliases|s|t|i|n|d|a|al"
)

# matches `\", \"key"` or `\", "key"` where the LLM failed to close a string value
# before a comma and the next key. inserts the missing closing quote.
_UNCLOSED_STRING_BEFORE_KEY = re.compile(rf'\\",(\s*)\\?"({_SEGMENT_KEYS})"(\s*):')

# trailing commas in arrays/objects (common LLM error)
_TRAILING_COMMA = re.compile(r",(\s*[\]}])")


def _repair_json(content: str) -> str:
    """apply targeted fixes for common LLM JSON malformations."""
    content = _UNCLOSED_STRING_BEFORE_KEY.sub(r'\\"",\1"\2"\3:', content)
    content = _TRAILING_COMMA.sub(r"\1", content)
    return content


def _json_error_snippet(content: str, pos: int, radius: int = 60) -> str:
    """return content around error position with a caret marker."""
    start = max(0, pos - radius)
    end = min(len(content), pos + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(content) else ""
    snippet = content[start:end].replace("\n", "\\n")
    caret_col = len(prefix) + (pos - start)
    return f"{prefix}{snippet}{suffix}\n{' ' * caret_col}^"


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
        pass

    # repair common LLM malformations, then try again (including trailing garbage)
    repaired = _repair_json(content)
    try:
        return cast(dict | list, json.loads(repaired))
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(repaired)
        return cast(dict | list, obj)


def _call_llm(
    messages: List[dict[str, str]],
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    seed: int = DEFAULT_SEED,
) -> str:
    """send messages to LLM and return raw content string. retries on API errors."""
    from .utils import log

    url = (
        f"{api_base}/chat/completions"
        if api_base
        else "https://api.openai.com/v1/chat/completions"
    )

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if thinking_budget > 0:
        body["thinking_budget_tokens"] = thinking_budget
    if seed > 0:
        body["seed"] = seed

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _call():
        log(
            "LLM_REQUEST",
            f"model={model}",
            {"messages": str(messages[-1])},
        )

        req_data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=req_data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
                res = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode(errors="replace")
            raise RuntimeError(f"llm http {e.code}: {error_body}") from e

        choices = res.get("choices", [])
        if not choices:
            log("LLM_ERROR", f"model={model}", {"response": str(res)})
            raise RuntimeError(f"llm returned no choices: {res}")

        choice = choices[0]
        content = choice.get("message", {}).get("content", "")
        if not content:
            diag = {
                "finish_reason": choice.get("finish_reason"),
                "message": str(choice.get("message")),
                "usage": str(res.get("usage")),
            }
            log("LLM_ERROR", f"model={model} empty content", diag)
            raise RuntimeError(
                f"llm returned empty content (finish_reason={diag['finish_reason']})"
            )

        # capture reasoning tokens from openai-compatible fields (reasoning_content,
        # reasoning) and from inline <think>/<reasoning> blocks in content. useful
        # for diagnosing retries where the model is "thinking" itself wrong.
        msg = choice.get("message", {}) or {}
        reasoning = (
            msg.get("reasoning_content")
            or msg.get("reasoning")
            or _extract_inline_reasoning(content)
            or ""
        )
        fields: dict[str, str] = {"response": content}
        if reasoning:
            fields["reasoning"] = reasoning
        log("LLM_RESPONSE", f"model={model}", fields)
        result: str = content
        return result

    response: str = retry_with_backoff(_call)
    return response


def _feedback_for_error(
    content: str, err: Exception, expected_shape: str | None = None
) -> str:
    """build targeted feedback for parse/validation errors to send back to LLM."""
    shape_hint = f" Expected shape: {expected_shape}" if expected_shape else ""
    if isinstance(err, json.JSONDecodeError):
        return (
            f"JSON parse error at line {err.lineno} col {err.colno}: {err.msg}.\n"
            f"Offending region (^ marks the error):\n"
            f"{_json_error_snippet(content, err.pos)}\n"
            f"Re-emit the entire response as valid JSON.{shape_hint}"
        )
    if isinstance(err, KeyError):
        return (
            f"Structure error: missing key {err}. "
            f"Re-emit with all required keys.{shape_hint}"
        )
    return f"Structure error: {err}. Re-emit with the correct shape.{shape_hint}"


def _query_llm_validated(
    messages: List[dict[str, str]],
    parse_fn: Callable[[dict | list], T],
    *,
    validate_fn: Callable[[T], list[str]] | None = None,
    model: str = DEFAULT_LLM_MODEL,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    max_retries: int = VALIDATION_MAX_RETRIES,
    label: str = "query",
    expected_shape: str | None = None,
) -> T:
    """query LLM, parse, validate; retry with targeted feedback on failure.

    mutates `messages` by appending each assistant response and feedback turn."""
    total = max_retries + 1
    for attempt in range(1, total + 1):
        content = _call_llm(messages, model, api_base, api_key, thinking_budget)
        feedback: str | None = None

        try:
            data = _parse_json_response(content)
            parsed = parse_fn(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            feedback = _feedback_for_error(content, e, expected_shape)
        else:
            errors = validate_fn(parsed) if validate_fn else []
            if not errors:
                if attempt > 1:
                    print(f"  {label}: attempt {attempt}/{total}: passed")
                return parsed
            feedback = (
                "Validation errors:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\nFix these and re-emit the full JSON."
            )

        summary = feedback.splitlines()[0]
        if attempt >= total:
            print(f"  {label}: attempt {attempt}/{total}: {summary}; giving up")
            raise ValueError(f"{label} failed after {max_retries} attempts: {feedback}")

        print(f"  {label}: attempt {attempt}/{total}: {summary}; sending feedback...")
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": feedback})

    raise RuntimeError("unreachable")


def _query_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    wrapper_keys: List[str] | None = None,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
    seed: int = DEFAULT_SEED,
) -> dict | list:
    """query LLM and return parsed JSON. for simple non-validated queries."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    content = _call_llm(messages, model, api_base, api_key, thinking_budget, seed)
    data = _parse_json_response(content)

    if wrapper_keys:
        if isinstance(data, dict):
            for key in wrapper_keys:
                if key in data:
                    unwrapped: dict | list = data[key]
                    return unwrapped
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            for key in wrapper_keys:
                if key in data[0]:
                    unwrapped_nested: dict | list = data[0][key]
                    return unwrapped_nested
    return data


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
        f"\nExisting characters (omit unless updating):\n{existing_cast_summary}\n"
        if existing_cast_summary
        else ""
    )

    prompt = f"""Identify book characters. Output a JSON object with a single \
key "characters" whose value is a list of character definitions.

{context_str}

For each character (new OR updated) output a definition with the following:

- name: Full canonical name
- description: A voice-design prompt written as a short paragraph (2-4 \
sentences). Open with a compact backstory clause that MOTIVATES the voice \
— role, condition, or defining circumstance that a listener would hear in \
the delivery (e.g. a former soldier gone to seed, a centenarian sustained \
by serums, an addict slackened by a hypnotic). Then describe the voice \
itself: gender, age, pitch, speed, volume, accent, texture/timbre, \
clarity, fluency, emotion, tone, and the audible personality traits. \
Every backstory detail must pay for itself by explaining a vocal trait — \
skip anything that doesn't (plot twists, physical appearance unrelated \
to voice, relationships, goals). Ground every claim in the prose; do not \
invent unsupported traits.
- audition_line: A sample line for this character. It should be two full sentences long.
- aliases: EVERY alternate form the prose uses — nicknames, shortened forms, \
last-name-only references, first-name-only references, and any stylized variants. \
Scan the text and include each distinct form you see.

Example: {{"characters": [{{"name": "Mirabel Thatcher-Quinn", \
"description": "A burnt-out field medic in her late twenties, still \
running on triage reflexes and too much black coffee. Female voice with \
a moderately low pitch, deliberate conversational pace that clips into \
urgency under pressure, and a flat American Midwestern accent. Dry, \
sardonic tone with clear articulation; resilient but audibly worn \
personality.", \
"audition_line": "I don't belong here. Let's just go.", \
"aliases": ["Mirabel", "Mira", "Thatcher-Quinn"]}}]}}

Always emit the "characters" key, even if the list has zero or one entries.

If you discover substantive new information about an existing character \
(description updates OR additional aliases used in the new chapters that aren't \
yet listed), you MUST re-emit their full character definition with the expanded \
information.
"""

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text_sample},
    ]
    return _query_llm_validated(
        messages,
        _parse_cast_list,
        validate_fn=_validate_cast_list,
        model=model,
        api_base=api_base,
        api_key=api_key,
        thinking_budget=thinking_budget,
        label="cast",
        expected_shape='{"characters": [{"name": ..., "description": ..., '
        '"audition_line": ..., "aliases": [...]}, ...]}',
    )


_CAST_WRAPPER_KEYS = ["characters", "c", "cast", "updates", "result", "results"]
_CHARACTER_KEYS = {
    "name",
    "n",
    "description",
    "d",
    "audition_line",
    "a",
    "aliases",
    "al",
}


def _parse_cast_list(data: list | dict) -> List[Character]:
    """parse LLM response into Character list, handling wrapped or bare formats."""
    if isinstance(data, dict):
        # unwrap common list wrappers
        for key in _CAST_WRAPPER_KEYS:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        # single-character dict: wrap in list
        if isinstance(data, dict) and _CHARACTER_KEYS & set(data.keys()):
            data = [data]
        # dict keyed by character name: {"Tam": {...}, "Seth": {...}}
        elif isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            data = [{"name": k, **v} for k, v in data.items()]

    if not isinstance(data, list):
        raise ValueError(f"expected list of characters, got {type(data).__name__}")

    results = []
    for i, c in enumerate(data):
        if not isinstance(c, dict):
            raise ValueError(f"character {i}: expected object, got {type(c).__name__}")
        # support both full and abbreviated key names
        name = str(c.get("name", c.get("n", "")))
        if not name:
            raise KeyError(f"character {i}: missing 'name'")
        results.append(
            Character(
                name=name,
                description=str(c.get("description", c.get("d", ""))),
                audition_line=str(c.get("audition_line", c.get("a", ""))),
                aliases=c.get("aliases", c.get("al")),
            )
        )
    return results


def _validate_cast_list(characters: List[Character]) -> list[str]:
    """check that each character has the fields needed for voice design."""
    errors = []
    for i, c in enumerate(characters):
        if not c.description:
            errors.append(f"character {i} ({c.name}): missing 'description'")
        if not c.audition_line:
            errors.append(f"character {i} ({c.name}): missing 'audition_line'")
    return errors


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
    """convert a text chunk into script segments with validation and feedback."""
    cast_str = _format_cast_list(characters_list)

    system_prompt = f"""Convert the following text to JSON. Output ONLY valid JSON, no markdown.

[Character List]
{cast_str}

{SCRIPT_GENERATION_COMMON}
"""

    messages: List[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text_chunk},
    ]
    return _query_llm_validated(
        messages,
        _parse_script_segments,
        validate_fn=lambda segs: _validate_script_segments(segs, characters_list),
        model=model,
        api_base=api_base,
        api_key=api_key,
        thinking_budget=thinking_budget,
        label="script",
        expected_shape=SCRIPT_EXPECTED_SHAPE,
    )


def _display_name(c: Character) -> str:
    """pick the shortest usable form from name+aliases, so the LLM doesn't have
    to repeat long canonical names (e.g. 21-word joke names) on every segment."""
    candidates = [c.name] + [a for a in (c.aliases or []) if len(a) >= 2]
    return min(candidates, key=len)


def _format_cast_list(characters_list: List[Character]) -> str:
    """format cast list for LLM prompts. surfaces the shortest form first to
    minimize tokens when the LLM echoes the speaker on every segment."""
    cast_info = []
    for c in characters_list:
        short = _display_name(c)
        others = [n for n in [c.name, *(c.aliases or [])] if n != short]
        if others:
            cast_info.append(f"{short} (also: {'; '.join(others)})")
        else:
            cast_info.append(short)
    return "- " + "\n- ".join(cast_info)


def _parse_script_segments(data: list | dict) -> List[ScriptSegment]:
    """parse LLM response into ScriptSegment list, handling wrapped or bare formats."""
    # unwrap if needed: {"seg": [...]} or {"segments": [...]} or any other wrapper
    if isinstance(data, dict):
        # try common wrapper keys first
        for key in ["seg", "segments"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            # check if it's a single segment dict (has speaker/text keys)
            if "speaker" in data or "s" in data:
                data = [data]
            # otherwise find the first list value
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break

    if not isinstance(data, list):
        raise ValueError(f"expected list of segments, got {type(data).__name__}")

    results = []
    for i, s in enumerate(data):
        if not isinstance(s, dict):
            continue
        # support both full and abbreviated key names
        speaker = s.get("speaker", s.get("s"))
        text = s.get("text", s.get("t"))
        if not speaker or not text:
            missing = []
            if not speaker:
                missing.append("speaker")
            if not text:
                missing.append("text")
            preview = str(s)[:100] + "..." if len(str(s)) > 100 else str(s)
            raise KeyError(f"segment {i} missing keys {missing}: {preview}")
        instruction = s.get("instruction", s.get("i", "")) or ""
        results.append(
            ScriptSegment(
                speaker=speaker,
                text=text,
                instruction=instruction,
            )
        )
    return results


def _normalize_name(s: str) -> str:
    """normalize a speaker name for fuzzy matching: casefold, strip trailing
    punctuation, collapse internal whitespace."""
    s = s.strip().rstrip(".,;:")
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def _build_name_index(
    characters_list: List[Character],
) -> dict[str, str]:
    """map normalized name/alias -> short display name for direct lookup."""
    special = {"Narrator", "Extra Female", "Extra Male"} | RETAINED_SPEAKERS
    index: dict[str, str] = {_normalize_name(n): n for n in special}
    for c in characters_list:
        display = _display_name(c)
        index.setdefault(_normalize_name(c.name), display)
        for alias in c.aliases or []:
            index.setdefault(_normalize_name(alias), display)
    return index


def _resolve_unambiguous_substring(
    normalized: str, characters_list: List[Character]
) -> list[str]:
    """return display names whose normalized name/alias contains (or is contained
    by) the given normalized speaker. used as a fallback for shortforms."""
    matches: set[str] = set()
    for c in characters_list:
        candidates = [c.name] + list(c.aliases or [])
        for cand in candidates:
            cn = _normalize_name(cand)
            if cn == normalized or cn in normalized or normalized in cn:
                matches.add(_display_name(c))
                break
    return sorted(matches)


def fix_instructions_inplace(segments: List[ScriptSegment]) -> int:
    """reset invalid instructions to 'neutral'. returns count fixed."""
    fixed = 0
    for seg in segments:
        if seg.instruction and seg.instruction not in EMOTION_KEYS:
            seg.instruction = "neutral"
            fixed += 1
    return fixed


def resolve_speakers(
    segments: List[ScriptSegment], characters_list: List[Character]
) -> list[str]:
    """resolve each segment's speaker to a canonical cast name.

    tries exact match, then normalized match (case/punctuation insensitive),
    then unambiguous substring match. mutates seg.speaker to canonical name
    on success. returns error messages for unresolved or ambiguous cases."""
    index = _build_name_index(characters_list)
    canonical_names = set(index.values())
    errors: list[str] = []

    for i, seg in enumerate(segments):
        if seg.speaker in canonical_names:
            continue
        norm = _normalize_name(seg.speaker)
        if norm in index:
            seg.speaker = index[norm]
            continue
        matches = _resolve_unambiguous_substring(norm, characters_list)
        if len(matches) == 1:
            seg.speaker = matches[0]
            continue
        if len(matches) > 1:
            errors.append(
                f"segment {i}: ambiguous speaker '{seg.speaker}', "
                f"could be any of: {', '.join(matches)}"
            )
        else:
            errors.append(f"segment {i}: unknown speaker '{seg.speaker}'")
    return errors


def _group_errors_by_message(errors: list[str]) -> list[str]:
    """collapse repeated errors that differ only in segment index.

    'segment 3: unknown speaker X' + 'segment 11: unknown speaker X' ->
    'segments [3, 11]: unknown speaker X'."""
    groups: dict[str, list[int]] = {}
    other: list[str] = []
    for e in errors:
        m = re.match(r"segment (\d+): (.*)", e)
        if m:
            groups.setdefault(m.group(2), []).append(int(m.group(1)))
        else:
            other.append(e)
    out = []
    for msg, idxs in groups.items():
        if len(idxs) == 1:
            out.append(f"segment {idxs[0]}: {msg}")
        else:
            preview = ", ".join(str(i) for i in idxs[:10])
            more = f" (+{len(idxs) - 10} more)" if len(idxs) > 10 else ""
            out.append(f"segments [{preview}{more}]: {msg}")
    return out + other


def _validate_script_segments(
    segments: List[ScriptSegment], characters_list: List[Character]
) -> list[str]:
    """auto-fix what we can (instructions, fuzzy speakers), then report residual
    errors with the cast list attached so the LLM can recover."""
    fix_instructions_inplace(segments)
    errors = resolve_speakers(segments, characters_list)
    if not errors:
        return []
    grouped = _group_errors_by_message(errors)
    cast_hint = "Valid speakers (use EXACTLY as written): " + ", ".join(
        _valid_speaker_names(characters_list)
    )
    return grouped + [cast_hint]


def _valid_speaker_names(characters_list: List[Character]) -> list[str]:
    """return valid short display names (plus specials), for feedback hints."""
    names = ["Narrator", "Extra Female", "Extra Male", *sorted(RETAINED_SPEAKERS)]
    for c in characters_list:
        names.append(_display_name(c))
    return names


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
    """convert a missing text fragment into script segments with multi-turn validation."""
    cast_str = _format_cast_list(characters_list)

    system_prompt = f"""\
Convert ONLY the "MISSING TEXT" to JSON. No markdown.

[Character List]
{cast_str}

{SCRIPT_GENERATION_COMMON}
- CRITICAL: Output ONLY words from MISSING TEXT.
  Never include words from surrounding script segments.
- Use the surrounding script segments (JSON) to determine speaker/tone,
  but output must contain ONLY MISSING TEXT words.
"""

    user_content = f"""
--- SURROUNDING SCRIPT BEFORE (JSON, for context only) ---
{context_before}

--- MISSING TEXT (convert this to script segments) ---
{missing_text}
--- END MISSING TEXT ---

--- SURROUNDING SCRIPT AFTER (JSON, for context only) ---
{context_after}
"""

    messages: List[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return _query_llm_validated(
        messages,
        _parse_script_segments,
        validate_fn=lambda segs: _validate_script_segments(segs, characters_list),
        model=model,
        api_base=api_base,
        api_key=api_key,
        thinking_budget=thinking_budget,
        label="fix",
        expected_shape=SCRIPT_EXPECTED_SHAPE,
    )
