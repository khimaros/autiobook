"""llm integration for script and cast generation."""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar, cast

from .config import (
    AUDITION_SAMPLE_LINE,
    DEFAULT_LLM_MODEL,
    DEFAULT_THINKING_BUDGET,
    EMOTION_KEYS,
    LLM_MAX_RETRIES,
    LLM_RETRY_DELAY,
    LLM_TIMEOUT,
    RETAINED_SPEAKERS,
    VALIDATION_MAX_RETRIES,
    active_seed,
)
from .utils import is_speakable
from .utils import retry_with_backoff as backoff

T = TypeVar("T")


class EmptyResponseError(RuntimeError):
    """model returned no content, with its whole reply left in the reasoning.

    not an api error: the request went through and the backend answered. under
    a fixed seed the same body yields the same empty reply, so the conversation
    has to change before another send is worth anything."""


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = LLM_MAX_RETRIES,
    initial_delay: float = LLM_RETRY_DELAY,
) -> T:
    """retry a function with exponential backoff on API errors."""
    return backoff(
        fn,
        max_retries,
        initial_delay,
        retryable=lambda e: not isinstance(e, EmptyResponseError),
    )


@dataclass
class Character:
    name: str
    description: str  # who they are: role, condition, what a listener should sense
    aliases: list[str] | None = None  # alternate names for the same character
    # the prompt actually sent to VoiceDesign. kept separate from `description`
    # so it stays purely acoustic -- backstory prose in the design prompt
    # dilutes the traits the model is being asked to produce.
    voice: str = ""
    # user-supplied only: set by hand in characters.json or `design --text`.
    # the llm is told not to propose one and any it proposes is dropped on
    # parse, so this never churns on its own.
    audition_line: str = ""

    def voice_prompt(self) -> str:
        """design prompt for this character, falling back to the description.

        casts written before the split carry a combined blob in `description`;
        using it keeps their generated voices byte-identical.
        """
        return self.voice or self.description

    def audition_text(self, override: str | None = None) -> str:
        """text this character's base reference clip speaks.

        a run-wide --audition-line wins, then a line set for this character by
        hand, then the standing sample line so clips differ only in voice.
        """
        return override or self.audition_line or AUDITION_SAMPLE_LINE


@dataclass
class CastMerge:
    """llm-directed merge of previously-distinct cast entries into one."""

    into: str  # canonical name of the surviving character
    from_: list[str]  # names of characters to fold in (and remove)
    reason: str = ""  # llm-supplied rationale (for audit log)


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
- "Retained" also covers front matter that is not part of the work itself: praise \
blurbs, review quotes and their attribution lines (e.g. a line that is only a \
publication or critic's name, with or without a leading dash), series lists, and \
publisher copy. These are unquoted text but they are NOT narration.
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
    r"speaker|text|instruction|name|description|voice|aliases" r"|s|t|i|n|d|v|a|al"
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


def _parses_as_json(content: str) -> bool:
    """true if content yields json under the same parsing the caller will use."""
    try:
        _parse_json_response(content)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


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
    seed: int | None = None,
) -> str:
    """send messages to LLM and return raw content string. retries on API errors."""
    from .utils import log

    # resolved here rather than as a default: the workdir seed is not known
    # until the command line is parsed, well after this module is imported.
    if seed is None:
        seed = active_seed()

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
        msg = choice.get("message", {}) or {}
        content = msg.get("content", "")
        # capture reasoning tokens from openai-compatible fields (reasoning_content,
        # reasoning) and from inline <think>/<reasoning> blocks in content. useful
        # for diagnosing retries where the model is "thinking" itself wrong.
        reasoning = (
            msg.get("reasoning_content")
            or msg.get("reasoning")
            or _extract_inline_reasoning(content)
            or ""
        )

        if not content:
            # some models put the whole answer in the reasoning field and leave
            # content empty. if what landed there parses, take it rather than
            # discarding a usable reply -- parse and validation still judge it.
            if reasoning and _parses_as_json(reasoning):
                content = reasoning
            else:
                diag = {
                    "finish_reason": choice.get("finish_reason"),
                    "message": str(msg),
                    "usage": str(res.get("usage")),
                }
                log("LLM_ERROR", f"model={model} empty content", diag)
                raise EmptyResponseError(
                    f"llm returned empty content "
                    f"(finish_reason={diag['finish_reason']})"
                )
        fields: dict[str, str] = {"response": content}
        if reasoning:
            fields["reasoning"] = reasoning
        log("LLM_RESPONSE", f"model={model}", fields)
        result: str = content
        return result

    response: str = retry_with_backoff(_call)
    return response


def _feedback_for_empty(expected_shape: str | None = None) -> str:
    """feedback for a reply whose content was empty (answer left in reasoning)."""
    shape_hint = f" Expected shape: {expected_shape}" if expected_shape else ""
    return (
        "Your last reply had no content, only reasoning. Put the answer in "
        f"the reply itself as valid JSON.{shape_hint}"
    )


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
            f"Redo the original task and emit the entire response as valid "
            f"JSON. Do NOT treat this error message or your previous reply as "
            f"input to convert.{shape_hint}"
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
    seed: int | None = None,
) -> T:
    """query LLM, parse, validate; retry with targeted feedback on failure.

    mutates `messages` by appending each assistant response and feedback turn."""
    total = max_retries + 1
    for attempt in range(1, total + 1):
        content = ""
        feedback: str | None = None

        try:
            content = _call_llm(
                messages, model, api_base, api_key, thinking_budget, seed
            )
        except EmptyResponseError:
            # re-sending is pointless under a fixed seed, but the feedback turn
            # changes the conversation, so the next attempt is a real attempt.
            feedback = _feedback_for_empty(expected_shape)

        if feedback is None:
            try:
                parsed = parse_fn(_parse_json_response(content))
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
    seed: int | None = None,
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
    audition_lines: bool = False,
) -> tuple[List[Character], List[CastMerge]]:
    """analyze text to identify characters and generate voice descriptions.

    returns `(characters, merges)`. `merges` directs the caller to fold
    previously-distinct cast entries together when the llm determines two
    names in `existing_cast_summary` actually refer to the same character.

    `audition_lines` asks for a per-character sample line as well. the caller
    only applies it where none is set, so it cannot overwrite a hand-written
    one; without it every clip speaks the shared line.
    """
    line_field = (
        """
- audition_line: Two full sentences this character would plausibly say, used \
to render their reference voice clip. Natural spoken dialogue in their own \
register. Avoid proper nouns, numbers and invented spellings, which a speech \
model reads poorly."""
        if audition_lines
        else ""
    )
    line_rule = (
        "Emit ONLY those five fields."
        if audition_lines
        else """Emit ONLY those four fields. In particular do NOT invent an \
audition_line or sample line: every reference clip speaks a fixed sentence, \
so one would be discarded. Put the accent and delivery in "voice" instead — \
that is the text the voice model actually reads."""
    )
    line_example = (
        '"audition_line": "I have seen worse, and I have seen better. '
        'Let us get on with it.", '
        if audition_lines
        else ""
    )
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
- description: ONE sentence on who they are — role, condition, or defining \
circumstance that a listener would hear in the delivery (e.g. a former \
soldier gone to seed, a centenarian sustained by serums, an addict \
slackened by a hypnotic). Every detail must pay for itself by explaining a \
vocal trait; skip plot twists, appearance unrelated to voice, \
relationships, and goals. Ground every claim in the prose.
- voice: The voice-design prompt, and ONLY the voice. A single sentence of \
comma-separated traits, drawn from the dimensions the voice model reads and \
using ITS vocabulary wherever one fits:
  gender: male, female, neutral
  age: child, teenager, young adult, middle-aged, senior
  pitch: high, medium, low, slightly high, slightly low
  speed: fast, medium, slow, slightly fast, slightly slow
  timbre: resonant, crisp, husky, mellow, sweet, deep, powerful
  emotion: cheerful, calm, gentle, serious, lively, composed, soothing
  accent: always name one, taken from how the character's dialogue is \
written where it carries dialect, and otherwise from where they are from in \
the story
Cover every dimension. The listed words are the ones the voice model reads \
best, so reach for them first, but add a short distinguishing phrase of your \
own wherever they are too coarse to tell this character apart. A cast in \
which two people share a voice is a failed cast: check the existing \
characters above and make sure yours is audibly nobody else. Do NOT describe \
volume or loudness: the model does not read it. NO backstory, NO narrative, \
NO character name — those belong in description and dilute the design \
prompt.{line_field}
- aliases: Alternate forms the NARRATOR uses to refer to this character in \
narration — i.e. names that appear in speaker-attribution tags ("said X", \
"X replied") or in descriptive narration ("X walked in"). Include nicknames, \
shortened forms, last-name-only, first-name-only, and stylized variants that \
the narrator actually uses this way. EXCLUDE vocatives and terms of address \
spoken by other characters inside dialogue (e.g. "baby", "kid", "boyo", "boss", \
"old son", "friend", "mon"), generic role words that aren't used as a name by \
the narrator, and epithets another character invents in the moment. NEVER a \
pronoun — not "he", "him", "his", "she", "her", "it", "they", "them", nor any \
other. A pronoun refers to whoever was last mentioned, so it names nobody; \
listing one hands this character every line the script attributes to that \
word. If in doubt, ask: would a listener hear this word and know it refers to \
this specific character as the speaker or subject? If not, omit it.

Example: {{"characters": [{{"name": "Mirabel Thatcher-Quinn", \
"description": "A burnt-out field medic in her late twenties, still \
running on triage reflexes and too much black coffee.", \
"voice": "Female, young adult, slightly low pitch, slightly fast speed that \
clips into urgency under pressure, husky timbre, serious, flat American \
Midwestern accent.", \
{line_example}"aliases": ["Mirabel", "Mira", "Thatcher-Quinn"]}}]}}

{line_rule}

Always emit the "characters" key, even if the list has zero or one entries.

If you discover substantive new information about an existing character \
(description updates OR additional aliases used in the new chapters that aren't \
yet listed), you MUST re-emit their full character definition with the expanded \
information.

If you determine that two or more entries in the existing characters list \
actually refer to the SAME character (e.g. a full name and a sobriquet that \
were misclassified as separate), emit a top-level "merges" list alongside \
"characters". Each merge is an object with "into" (the surviving canonical \
name, chosen from the existing list), "from" (a list of other existing names \
to fold in — their aliases will be preserved on the survivor and they will \
be removed from the cast), and a short "reason". Only merge names that \
already appear in the existing characters list. Do not merge distinct \
characters who merely share a title or role. Omit "merges" entirely if no \
merges are needed.

Example with merges: {{"characters": [], "merges": [{{"into": "Mirabel \
Thatcher-Quinn", "from": ["The Medic"], "reason": "narration in chapter 4 \
reveals 'the medic' is Mirabel"}}]}}
"""

    expected_line = '"audition_line": ..., ' if audition_lines else ""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text_sample},
    ]
    return _query_llm_validated(
        messages,
        lambda data: _parse_cast_response(data, keep_audition_line=audition_lines),
        validate_fn=_validate_cast_response,
        model=model,
        api_base=api_base,
        api_key=api_key,
        thinking_budget=thinking_budget,
        label="cast",
        expected_shape='{"characters": [{"name": ..., "description": ..., "voice": ..., '
        f'{expected_line}"aliases": [...]}}, ...], '
        '"merges": [{"into": ..., "from": [...], "reason": ...}]}',
    )


_CAST_WRAPPER_KEYS = ["characters", "c", "cast", "updates", "result", "results"]
_CHARACTER_KEYS = {
    "name",
    "n",
    "description",
    "d",
    "voice",
    "v",
    "aliases",
    "al",
}


def _parse_cast_list(
    data: list | dict, keep_audition_line: bool = False
) -> List[Character]:
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
                aliases=c.get("aliases", c.get("al")),
                audition_line=(
                    str(c.get("audition_line", c.get("a", "")))
                    if keep_audition_line
                    else ""
                ),
                voice=str(c.get("voice", c.get("v", ""))),
            )
        )
    return results


def _parse_merges(data: list | dict) -> List[CastMerge]:
    """parse optional merges list from cast LLM response."""
    if isinstance(data, dict):
        raw = data.get("merges") or data.get("m") or []
    else:
        raw = []
    if not isinstance(raw, list):
        raise ValueError(f"expected list of merges, got {type(raw).__name__}")
    merges: List[CastMerge] = []
    for i, m in enumerate(raw):
        if not isinstance(m, dict):
            raise ValueError(f"merge {i}: expected object, got {type(m).__name__}")
        into = str(m.get("into", m.get("i", ""))).strip()
        if not into:
            raise KeyError(f"merge {i}: missing 'into'")
        from_raw = m.get("from", m.get("f", []))
        if isinstance(from_raw, str):
            from_raw = [from_raw]
        if not isinstance(from_raw, list):
            raise ValueError(f"merge {i}: 'from' must be a list")
        from_ = [str(x).strip() for x in from_raw if str(x).strip()]
        if not from_:
            raise ValueError(f"merge {i}: 'from' must be non-empty")
        merges.append(
            CastMerge(into=into, from_=from_, reason=str(m.get("reason", "")))
        )
    return merges


def _parse_cast_response(
    data: list | dict, keep_audition_line: bool = False
) -> tuple[List[Character], List[CastMerge]]:
    """parse the full cast LLM response into characters and optional merges."""
    merges = _parse_merges(data) if isinstance(data, dict) else []
    characters = _parse_cast_list(data, keep_audition_line=keep_audition_line)
    return characters, merges


def _validate_cast_response(
    parsed: tuple[List[Character], List[CastMerge]],
) -> list[str]:
    characters, merges = parsed
    errors = _validate_cast_list(characters)
    seen_from: dict[str, str] = {}
    for i, m in enumerate(merges):
        into_key = _normalize_name(m.into)
        if not into_key:
            errors.append(f"merge {i}: 'into' is empty after normalization")
            continue
        for src in m.from_:
            src_key = _normalize_name(src)
            if not src_key:
                errors.append(f"merge {i}: empty entry in 'from'")
                continue
            if src_key == into_key:
                errors.append(
                    f"merge {i}: 'from' entry {src!r} equals 'into' {m.into!r}"
                )
            if src_key in seen_from:
                errors.append(
                    f"merge {i}: {src!r} already folded into " f"{seen_from[src_key]!r}"
                )
            else:
                seen_from[src_key] = m.into
    return errors


def _validate_cast_list(characters: List[Character]) -> list[str]:
    """check that each character has the fields needed for voice design and
    that names/aliases don't collide across characters."""
    errors = []
    for i, c in enumerate(characters):
        if not c.description:
            errors.append(f"character {i} ({c.name}): missing 'description'")
        if not c.voice:
            errors.append(f"character {i} ({c.name}): missing 'voice'")

    # name/alias collisions (normalized) across characters
    owners: dict[str, str] = {}
    for c in characters:
        for label in [c.name, *(c.aliases or [])]:
            key = _normalize_name(label)
            if not key:
                continue
            if label != c.name and key == _normalize_name(c.name):
                errors.append(
                    f"character '{c.name}': alias '{label}' conflicts with its own name"
                )
                continue
            if key in owners and owners[key] != c.name:
                kind = "name" if label == c.name else "alias"
                errors.append(
                    f"character '{c.name}': {kind} '{label}' already used by "
                    f"'{owners[key]}'"
                )
            else:
                owners[key] = c.name
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
    seed: int | None = None,
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
        _parse_script_response,
        validate_fn=lambda segs: _validate_script_segments(segs, characters_list),
        model=model,
        api_base=api_base,
        api_key=api_key,
        thinking_budget=thinking_budget,
        label="script",
        expected_shape=SCRIPT_EXPECTED_SHAPE,
        seed=seed,
    )


def _display_name(c: Character) -> str:
    """pick the shortest usable form from name+aliases, so the LLM doesn't have
    to repeat long canonical names (e.g. 21-word joke names) on every segment."""
    candidates = [c.name] + [a for a in (c.aliases or []) if len(a) >= 2]
    return min(candidates, key=len)


def _format_cast_list(characters_list: List[Character], specials: bool = False) -> str:
    """format cast list for LLM prompts. surfaces the shortest form first to
    minimize tokens when the LLM echoes the speaker on every segment.

    `specials` appends the non-character speakers. review needs them: it is
    told to use names exactly as listed, so a speaker missing from the list
    reads as a mis-attribution to correct rather than a value to preserve.
    """
    cast_info = []
    for c in characters_list:
        short = _display_name(c)
        others = [n for n in [c.name, *(c.aliases or [])] if n != short]
        if others:
            cast_info.append(f"{short} (also: {'; '.join(others)})")
        else:
            cast_info.append(short)
    if specials:
        cast_info.append("Narrator (prose narration)")
        cast_info.append("Extra Female / Extra Male (unnamed minor characters)")
        cast_info.append(
            f"{' / '.join(sorted(RETAINED_SPEAKERS))} (kept in the text, never spoken)"
        )
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


def merge_unspeakable_segments(segments: List[ScriptSegment]) -> List[ScriptSegment]:
    """fold a wordless segment into a neighbour sharing its speaker.

    where narration interrupts dialogue, the model reliably emits the
    resumption quote as a segment of its own ("One would think," / Tresting
    noted, / the bare quote / "that a thousand years..."), which has nothing to
    perform. it belongs to the line it opens, so the next segment is preferred
    and the previous one takes a closing mark. text is only moved, never
    rewritten, so the concatenation still matches the source.
    """
    out: List[ScriptSegment] = []
    carry = ""
    for i, seg in enumerate(segments):
        text = carry + seg.text
        carry = ""
        if is_speakable(seg.text):
            out.append(ScriptSegment(seg.speaker, text, seg.instruction))
        elif i + 1 < len(segments) and segments[i + 1].speaker == seg.speaker:
            carry = text
        elif out and out[-1].speaker == seg.speaker:
            prev = out[-1]
            out[-1] = ScriptSegment(prev.speaker, prev.text + text, prev.instruction)
        else:
            # no neighbour to take it; perform drops it rather than synthesizing
            out.append(ScriptSegment(seg.speaker, text, seg.instruction))
    return out


def _parse_script_response(data: list | dict) -> List[ScriptSegment]:
    """parse generated script segments, folding away the wordless ones.

    the split path parses without this: its parts must stay as the model drew
    them for the concatenation and minimum-count checks.
    """
    return merge_unspeakable_segments(_parse_script_segments(data))


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


REVIEW_CHANGES_SHAPE = (
    '{"changes": [{"index": ..., "speaker": ..., "instruction": ...}, ...], '
    '"flags": [{"index": ..., "speaker": ..., "instruction": ..., "reason": ...}, ...]}'
)


@dataclass
class ReviewFlag:
    """LLM-flagged segment needing human attention (e.g. needs splitting).

    suggestion: optional best-guess correction (speaker/instruction only) the
    human can [a]pply via the audit walkthrough. text is never suggested —
    text correctness is verified separately by `revise`.
    """

    index: int
    reason: str
    suggestion: Optional[dict] = None


def _extract_flags_list(data: list | dict) -> list:
    """pull the optional flags list out of the LLM response."""
    if isinstance(data, dict):
        v = data.get("flags")
        if isinstance(v, list):
            return v
    return []


def _extract_changes_list(data: list | dict) -> list:
    """unwrap various LLM response shapes to a flat list of change objects."""
    if isinstance(data, dict):
        for key in ("changes", "segments", "seg"):
            v = data.get(key)
            if isinstance(v, list):
                return v
        if "index" in data:
            return [data]
        for v in data.values():
            if isinstance(v, list):
                return v
        return []
    if isinstance(data, list):
        return data
    raise ValueError(f"expected change list, got {type(data).__name__}")


def _apply_review_changes(original: List[ScriptSegment], changes: list[dict]) -> tuple[
    List[ScriptSegment],
    list[tuple[int, str]],
    list[tuple[int, str]],
    list[tuple[int, str]],
]:
    """merge a sparse changes list onto a copy of the original batch by index.

    returns (merged, text_mutations, invalid_instructions, retained_edits).
    text_mutations records (idx, attempted_text) for any change that tried to
    alter segment text — used to detect when the LLM hallucinates during
    review. text is always preserved on disk regardless. invalid_instructions
    records (idx, attempted_instruction) for changes whose instruction was not
    in EMOTION_KEYS. retained_edits records (idx, attempted_speaker) for
    changes that tried to voice a retained segment. both are ignored and the
    original kept.
    """
    merged = list(original)
    text_mutations: list[tuple[int, str]] = []
    invalid_instructions: list[tuple[int, str]] = []
    retained_edits: list[tuple[int, str]] = []
    for c in changes:
        if not isinstance(c, dict):
            continue
        idx = c.get("index", c.get("i"))
        if not isinstance(idx, int) or idx < 0 or idx >= len(merged):
            preview = str(c)[:100] + "..." if len(str(c)) > 100 else str(c)
            raise KeyError(f"change missing valid index: {preview}")
        cur = merged[idx]
        speaker = c.get("speaker", c.get("s", cur.speaker)) or cur.speaker
        # retained text is not narration: section markers, chapter numbers and
        # front matter (blurbs and their attribution lines). the prompt says so,
        # but a reviewer reading "Narrator for ALL unquoted text" reclassifies
        # them anyway, and the result is a voice reading "-- Financial Times".
        if cur.speaker in RETAINED_SPEAKERS and speaker not in RETAINED_SPEAKERS:
            print(
                f"  review: ignoring attempt to voice retained segment {idx} "
                f"as {speaker!r}, keeping {cur.speaker!r}"
            )
            retained_edits.append((idx, str(speaker)))
            speaker = cur.speaker
        instruction = c.get("instruction", c.get("in", cur.instruction))
        if instruction is None:
            instruction = cur.instruction
        # skip unrecognized instructions rather than letting validation silently
        # collapse them to "neutral" — the LLM often re-emits an instruction when
        # only fixing the speaker, and any synonym or variant would wipe a
        # perfectly good original.
        if instruction and instruction not in EMOTION_KEYS:
            print(
                f"  review: ignoring invalid instruction "
                f"{instruction!r} at segment {idx}, keeping {cur.instruction!r}"
            )
            invalid_instructions.append((idx, str(instruction)))
            instruction = cur.instruction
        emitted_text = c.get("text", c.get("t"))
        if isinstance(emitted_text, str) and emitted_text.strip() != cur.text.strip():
            text_mutations.append((idx, emitted_text))
        # text is NOT reviewable; always preserve the original to prevent the
        # LLM from truncating, rewording, or otherwise degrading source-faithful text.
        merged[idx] = ScriptSegment(
            speaker=speaker, text=cur.text, instruction=instruction or ""
        )
    return merged, text_mutations, invalid_instructions, retained_edits


def review_script_batch(
    source_span: str,
    segments: List[ScriptSegment],
    characters_list: List[Character],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> tuple[
    List[ScriptSegment],
    list[ReviewFlag],
    list[tuple[int, str]],
    list[tuple[int, str]],
    list[tuple[int, str]],
]:
    """review a batch of segments against the covering source text.

    returns (corrected_segments, flags, text_mutations, invalid_instructions,
    retained_edits).
    text is never modified; the LLM emits speaker/instruction corrections by
    index and may flag segments needing human attention (e.g. a segment that
    should be split). text_mutations captures any LLM attempt to alter segment
    text (defensive — text is preserved on disk regardless, but the attempt is
    logged so reviewers can see it). invalid_instructions captures any change
    whose instruction was not in EMOTION_KEYS; such corrections are ignored
    and the original instruction is kept."""
    cast_str = _format_cast_list(characters_list, specials=True)
    current_json = json.dumps(
        [
            {
                "index": i,
                "speaker": s.speaker,
                "text": s.text,
                "instruction": s.instruction,
            }
            for i, s in enumerate(segments)
        ],
        ensure_ascii=False,
    )

    system_prompt = f"""\
Review script segments against the SOURCE TEXT and emit ONLY the segments whose \
SPEAKER attribution or INSTRUCTION is wrong.

[Character List]
{cast_str}

{SCRIPT_GENERATION_COMMON}

Review Rules:

- Output a JSON object {{"changes": [...]}} containing ONLY segments you are \
correcting. Omit any segment that is already correct.
- Each change MUST include the "index" field copied verbatim from the input, \
plus the corrected "speaker" and "instruction".
- DO NOT modify the "text" field. Text is not under review. You do not need to \
emit "text" at all; if you do, it MUST be byte-for-byte identical to the input.
- If no changes are needed, return {{"changes": []}}.
- Do NOT renumber, reorder, split, or merge segments. Only speaker/instruction \
corrections.
- Fix SPEAKER attribution errors (narration tagged as a character, or the wrong \
character speaking dialogue).
- Fix INSTRUCTION values that don't match the tone of the dialogue or the narration.
- Do NOT alter "Retained" segments.

Human Review Flags:

- Flag ANY uncertainty. If you are not confident about the speaker, the \
instruction, or whether a segment is structured correctly, raise a flag rather \
than guessing. It is FAR better to surface a doubt for human review than to \
silently apply a wrong correction. Examples worth flagging: speaker is \
ambiguous from context (could plausibly be more than one character), \
attribution is missing in source so the speaker has to be inferred, the cast \
list does not contain an obvious match for the speaker, dialogue and narration \
appear merged into one segment and should be split, or two distinct source \
elements (e.g. a chapter heading and the following paragraph) are merged into \
one segment.
- A flag has the SAME shape as a change, plus a "reason" field: \
{{"index": <int>, "speaker": "...", "instruction": "...", "reason": "<short \
note>"}} on a top-level "flags" list. "speaker" and "instruction" are your \
best-guess suggestion (optional — omit either if you have no suggestion); the \
human reviews and applies them. Omit "flags" entirely if nothing needs human \
attention.
- The ONLY difference between "changes" and "flags" is confidence. Confident \
corrections go in "changes" and are auto-applied; uncertain ones go in \
"flags" with the same fields plus a "reason", and wait for human review. Do \
not put the same segment in both lists.
- DO NOT flag for text issues. Trust the segment text. Text-vs-source coverage \
(missing fragments, hallucinations, garbled or extraneous content) is verified \
and repaired by a separate `revise` phase — duplicating that here just adds \
noise. Even if a segment's text looks wrong, garbled, or absent from the \
source, do NOT flag it.
"""

    user_content = f"""\
--- SOURCE TEXT (authoritative) ---
{source_span}
--- END SOURCE TEXT ---

--- CURRENT SCRIPT SEGMENTS (JSON; each has an "index"; review and correct) ---
{current_json}
--- END CURRENT SCRIPT SEGMENTS ---
"""

    captured_flags: list[ReviewFlag] = []
    captured_mutations: list[tuple[int, str]] = []
    captured_invalid_instructions: list[tuple[int, str]] = []
    captured_retained_edits: list[tuple[int, str]] = []

    def parse_changes(data: list | dict) -> List[ScriptSegment]:
        captured_flags.clear()
        captured_mutations.clear()
        captured_invalid_instructions.clear()
        captured_retained_edits.clear()
        for f in _extract_flags_list(data):
            if not isinstance(f, dict):
                continue
            idx = f.get("index", f.get("i"))
            reason = f.get("reason") or f.get("r") or ""
            if not isinstance(idx, int) or not (0 <= idx < len(segments)):
                continue
            cur = segments[idx]
            # flags share the change shape (speaker/instruction at top level);
            # also accept legacy nested "suggestion" form.
            src = f["suggestion"] if isinstance(f.get("suggestion"), dict) else f
            emitted_text = src.get("text", src.get("t"))
            if (
                isinstance(emitted_text, str)
                and emitted_text.strip() != cur.text.strip()
            ):
                captured_mutations.append((idx, emitted_text))
            speaker = src.get("speaker", src.get("s"))
            instruction = src.get("instruction", src.get("in"))
            # drop invalid instruction suggestions and echoes of the current
            # value — otherwise the reviewer sees a "suggestion" that's
            # identical to the on-disk segment and has nothing to apply.
            if isinstance(instruction, str) and instruction not in EMOTION_KEYS:
                instruction = None
            if isinstance(speaker, str) and speaker == cur.speaker:
                speaker = None
            if isinstance(instruction, str) and instruction == cur.instruction:
                instruction = None
            suggestion: Optional[dict] = None
            if speaker is not None or instruction is not None:
                suggestion = {
                    "speaker": speaker or cur.speaker,
                    "instruction": (
                        instruction if instruction is not None else cur.instruction
                    ),
                }
            captured_flags.append(
                ReviewFlag(index=idx, reason=str(reason), suggestion=suggestion)
            )
        merged, mutations, invalid_instructions, retained_edits = _apply_review_changes(
            segments, _extract_changes_list(data)
        )
        captured_mutations.extend(mutations)
        captured_invalid_instructions.extend(invalid_instructions)
        captured_retained_edits.extend(retained_edits)
        return merged

    messages: List[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    corrected = _query_llm_validated(
        messages,
        parse_changes,
        validate_fn=lambda segs: _validate_script_segments(segs, characters_list),
        model=model,
        api_base=api_base,
        api_key=api_key,
        thinking_budget=thinking_budget,
        label="review",
        expected_shape=REVIEW_CHANGES_SHAPE,
    )
    return (
        corrected,
        list(captured_flags),
        list(captured_mutations),
        list(captured_invalid_instructions),
        list(captured_retained_edits),
    )


_NORM_TEXT_RE = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    """collapse all whitespace for text-faithfulness comparison."""
    return _NORM_TEXT_RE.sub("", s)


def split_mixed_segment(
    segment: ScriptSegment,
    context_before: str,
    context_after: str,
    characters_list: List[Character],
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_LLM_MODEL,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> tuple[List[ScriptSegment], Optional[str]]:
    """split a segment that mixes narration and dialogue into multiple segments.

    text-preserving: the concatenation of returned segments' text must equal
    the input segment's text exactly (whitespace differences are tolerated).

    returns (segments, flag_reason). if the LLM disputes the split request
    (e.g. believes the segment is not actually mixed), it can respond with a
    flag instead of segments — flag_reason carries the disagreement text and
    segments is empty. caller is expected to surface the flag for human review.
    """
    cast_str = _format_cast_list(characters_list)

    system_prompt = f"""\
Split the INPUT SEGMENT below at NARRATION/DIALOGUE BOUNDARIES ONLY. The \
concatenation of your output segments' "text" MUST exactly equal the INPUT \
SEGMENT's "text", word-for-word and in the same order. Do NOT add, remove, \
reword, or reorder any text — only SPLIT it.

[Character List]
{cast_str}

{SCRIPT_GENERATION_COMMON}

Split Rules (these OVERRIDE any general guidance about segment length or \
sentence count):

- Split ONLY where the speaker changes — i.e. between narration prose and a \
character's quoted dialogue, or between two different characters' quoted \
lines. These are the only legitimate split points.
- DO NOT split for length, readability, sentence count, or style. A long \
narration block stays as ONE segment. A long single-speaker quote stays as \
ONE segment. Length is NOT a reason to split.
- DO NOT split mid-sentence, mid-clause, or anywhere inside a single speaker's \
continuous text. The 2–3-sentence-per-segment guidance above does NOT apply \
here — your job is structural repair, not stylistic re-segmentation.
- CRITICAL: Output text MUST be a faithful split of the INPUT SEGMENT text \
ONLY. Never include words from the surrounding context segments.
- Use the surrounding script segments to determine speaker attribution and \
instruction for each part, but output text comes ONLY from the INPUT SEGMENT.

If you genuinely believe the INPUT SEGMENT does NOT contain a \
narration/dialogue boundary and should NOT be split (e.g. the quoted text is \
a single speaker's full line with no narration, the "quotes" are scare \
quotes / titles / not actual dialogue, the entire segment is one continuous \
voice, or you cannot identify a faithful split), respond with this shape \
instead of "segments":

```
{{"flag": {{"reason": "<short explanation of why this should not be split>"}}}}
```

This pushes the segment to a human reviewer rather than forcing a split. \
Prefer flagging over guessing or splitting for length.
"""

    input_json = json.dumps(
        {
            "speaker": segment.speaker,
            "text": segment.text,
            "instruction": segment.instruction,
        },
        ensure_ascii=False,
    )

    user_content = f"""
--- SURROUNDING SCRIPT BEFORE (JSON, for context only) ---
{context_before}

--- INPUT SEGMENT (split this into multiple segments; text concatenation MUST match) ---
{input_json}
--- END INPUT SEGMENT ---

--- SURROUNDING SCRIPT AFTER (JSON, for context only) ---
{context_after}
"""

    expected = _normalize_text(segment.text)

    def parse_split(data: list | dict) -> tuple[List[ScriptSegment], Optional[str]]:
        if isinstance(data, dict) and isinstance(data.get("flag"), dict):
            reason = data["flag"].get("reason") or "no reason given"
            return [], str(reason)
        return _parse_script_segments(data), None

    def validate_split(
        result: tuple[List[ScriptSegment], Optional[str]],
    ) -> list[str]:
        segs, flag = result
        if flag is not None:
            return []  # disagreement flag is always accepted
        errors = _validate_script_segments(segs, characters_list)
        if len(segs) < 2:
            errors.append(
                "must produce at least 2 segments (the input was a single "
                "segment that needs SPLITTING). if you believe no split is "
                'needed, respond with {"flag": {"reason": "..."}} instead.'
            )
        joined = _normalize_text("".join(s.text for s in segs))
        if joined != expected:
            errors.append(
                f"text concatenation does not match input segment "
                f"(got {len(joined)} non-ws chars, expected {len(expected)}). "
                f"output text MUST be a faithful split of input text only."
            )
        return errors

    messages: List[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return _query_llm_validated(
        messages,
        parse_split,
        validate_fn=validate_split,
        model=model,
        api_base=api_base,
        api_key=api_key,
        thinking_budget=thinking_budget,
        label="split",
        expected_shape=SCRIPT_EXPECTED_SHAPE,
    )


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
        _parse_script_response,
        validate_fn=lambda segs: _validate_script_segments(segs, characters_list),
        model=model,
        api_base=api_base,
        api_key=api_key,
        thinking_budget=thinking_budget,
        label="fix",
        expected_shape=SCRIPT_EXPECTED_SHAPE,
    )
