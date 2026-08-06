"""shared constants and configuration."""

import os
import random
import re

# epub parsing
CONTENT_TAGS = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"]
SKIP_TAGS = [
    "script",
    "style",
    "meta",
    "head",
    "link",
    "noscript",
    "nav",
    "header",
    "footer",
]
MIN_CHAPTER_WORDS = 50

# tts settings
DEFAULT_MODEL = os.getenv(
    "AUTIOBOOK_TTS_INSTRUCT_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
)
VOICE_DESIGN_MODEL = os.getenv(
    "AUTIOBOOK_TTS_DESIGN_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)
BASE_MODEL = os.getenv("AUTIOBOOK_TTS_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
# one model for every mode, overriding the three qwen-specific defaults above.
# hosted providers serve design, cloning and synthesis from a single model id.
DEFAULT_TTS_MODEL = os.getenv("AUTIOBOOK_TTS_MODEL", "")
DEFAULT_SPEAKER = "ryan"
RETAINED_SPEAKERS = {"Retained", "Unvoiced", "Silent"}  # text kept but not narrated
MAX_CHUNK_SIZE = 500  # balance between coherence and decode speed
SAMPLE_RATE = 24000

# llm settings
DEFAULT_LLM_MODEL = os.getenv("AUTIOBOOK_LLM_MODEL", "openai/gpt-4o")
# review uses a stricter pass over already-generated scripts; defaults to the
# main llm model but can be overridden separately for cost/quality tuning.
DEFAULT_REVIEW_LLM_MODEL = os.getenv("AUTIOBOOK_REVIEW_LLM_MODEL", DEFAULT_LLM_MODEL)
DEFAULT_THINKING_BUDGET = int(os.getenv("AUTIOBOOK_LLM_THINKING_BUDGET", "16384"))
LLM_TIMEOUT = int(os.getenv("AUTIOBOOK_LLM_TIMEOUT", "600"))
LLM_MAX_RETRIES = 3
VALIDATION_MAX_RETRIES = 5
LLM_RETRY_DELAY = 1.0  # initial delay in seconds, doubles on each retry
CAST_BATCH_SIZE = int(os.getenv("AUTIOBOOK_CAST_BATCH_SIZE", "3"))
# review command sends this many consecutive script segments per LLM call along
# with the covering source-text span so the model can correct mis-attributions
# and other drift (text, emotion) against the original.
REVIEW_BATCH_SIZE = int(os.getenv("AUTIOBOOK_REVIEW_BATCH_SIZE", "50"))
# cast generation chunks chapters into windows this many words wide so the
# LLM has enough coreference context (pronouns, dialogue attribution) to
# identify speakers — much larger than script-writing chunks (~1500 words)
# because identity resolution benefits from broader span than transcription.
CAST_CHUNK_WORDS = int(os.getenv("AUTIOBOOK_CAST_CHUNK_WORDS", "4000"))
# overlap (words) between successive cast chunks so a character introduced
# at the tail of chunk N remains recognizable in chunk N+1.
CAST_CHUNK_OVERLAP_WORDS = int(os.getenv("AUTIOBOOK_CAST_CHUNK_OVERLAP_WORDS", "400"))

# seed for reproducibility (tts + llm). unset → generate one concrete random
# seed per process so the exact value can be logged and recorded with output.
# a workdir's resolved seed is persisted to SEED_FILE and reused on later runs,
# so resuming a book stays reproducible without pinning AUTIOBOOK_SEED by hand.
_seed_env = os.getenv("AUTIOBOOK_SEED")
DEFAULT_SEED = int(_seed_env) if _seed_env else random.randint(1, 2**31 - 1)
SEED_FILE = "seed.json"

_active_seed: int | None = None


def active_seed() -> int:
    """seed in effect for this process.

    resolved from the workdir once the command line is parsed; falls back to
    the import-time default for commands that have no workdir. consumers read
    it lazily (default_factory / None sentinel) because module-level defaults
    would bind before the workdir is known.
    """
    return DEFAULT_SEED if _active_seed is None else _active_seed


def set_active_seed(seed: int) -> None:
    """pin the seed for the remainder of this process."""
    global _active_seed
    _active_seed = seed


# tts http settings
TTS_HTTP_TIMEOUT = int(os.getenv("AUTIOBOOK_TTS_TIMEOUT", "300"))
# tts backend dialects. "qwen" is the local qwen3-tts server: wav responses,
# sse streaming, /audio/voices for listing and cloning, and sampler fields.
# "openai" is the strict subset hosted providers accept (openrouter, openai):
# model/input/voice/response_format/instructions answered as raw audio bytes.
# "auto" picks by host, since a hosted endpoint rejects the qwen extensions
# and bills for the attempt.
TTS_DIALECT_AUTO = "auto"
TTS_DIALECT_QWEN = "qwen"
TTS_DIALECT_OPENAI = "openai"
TTS_DIALECTS = [TTS_DIALECT_AUTO, TTS_DIALECT_QWEN, TTS_DIALECT_OPENAI]
DEFAULT_TTS_DIALECT = os.getenv("AUTIOBOOK_TTS_DIALECT", TTS_DIALECT_AUTO)
OPENAI_DIALECT_HOSTS = ["openrouter.ai", "api.openai.com"]
# speech response formats. openrouter serves pcm (its default) or mp3 and
# never wav; the local server serves wav. "" resolves from the dialect.
TTS_FORMAT_WAV = "wav"
TTS_FORMAT_PCM = "pcm"
TTS_FORMAT_MP3 = "mp3"
TTS_RESPONSE_FORMATS = [TTS_FORMAT_WAV, TTS_FORMAT_PCM, TTS_FORMAT_MP3]
DEFAULT_TTS_RESPONSE_FORMAT = os.getenv("AUTIOBOOK_TTS_RESPONSE_FORMAT", "")
# preset voice names for backends with no /audio/voices endpoint. openrouter
# publishes no voice discovery api, so --preset-voices needs them supplied.
TTS_PRESET_VOICES = [
    v.strip() for v in os.getenv("AUTIOBOOK_TTS_VOICES", "").split(",") if v.strip()
]
# how delivery direction reaches the model. "field" sends a top-level
# instructions field, which the qwen server and openai's own api read.
# openrouter documents no such field, so "prefix" folds the direction into the
# input text instead -- how gemini tts takes direction, and the only channel
# left on a provider that drops unknown fields.
TTS_DIRECTION_FIELD = "field"
TTS_DIRECTION_PREFIX = "prefix"
TTS_DIRECTIONS = [TTS_DIRECTION_FIELD, TTS_DIRECTION_PREFIX]
DEFAULT_TTS_DIRECTION = os.getenv("AUTIOBOOK_TTS_DIRECTION", TTS_DIRECTION_FIELD)
TTS_DIRECTION_TEMPLATE = (
    "Read the following aloud. Voice and delivery: {instruct}\n\n{text}"
)
# streaming audio batch size for the qwen server. 0 disables streaming (single
# response_format=wav reply). >0 enables SSE per-batch PCM streaming so playback
# starts before synthesis finishes. 16 frames ≈ 1.28s audio; range 8-32.
# hosted providers ignore this: their response body is already a byte stream.
TTS_STREAM_BATCH_SIZE = int(os.getenv("AUTIOBOOK_TTS_STREAM_BATCH_SIZE", "0"))

# audio processing. PARAGRAPH_PAUSE_MS is the single source of truth for the
# silence inserted between chunks: chapter assembly and the timing manifest must
# use the same value or every cue/overlay offset drifts by 500ms per chunk.
PARAGRAPH_PAUSE_MS = 500
CHAPTER_PAUSE_MS = 1000

# leading dashes are typographic, not phonetic: a blurb attribution line reads
# as the publication's name alone. some tts models answer the bare dash with
# silence, so it is stripped before synthesis. dashes inside a line are left
# alone; they shape prosody. covers ascii hyphen, U+2010-U+2015, minus sign.
LEADING_DASH_RE = re.compile(r"^[\s\-\u2010-\u2015\u2212]+")

# mp3 export
DEFAULT_BITRATE = "192k"
UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# logging
LOG_FILE = "autiobook.log"

# epub3 media overlay export
EPUB_EXT = ".epub"
SMIL_EXT = ".smil"
OVERLAY_DIR = "autiobook"  # in-epub folder for generated smil + audio
OVERLAY_ID_PREFIX = "aob"  # generated fragment ids, e.g. id="aob12"
# reserved epub3 vocabulary; readers apply this class to the active fragment.
OVERLAY_ACTIVE_CLASS = "-epub-media-overlay-active"
OVERLAY_HIGHLIGHT_CSS = (
    f".{OVERLAY_ACTIVE_CLASS} {{ background: rgba(255, 214, 0, 0.35); }}"
)
NAV_FILE = "nav.xhtml"
# embedded audio dominates read-along epub size, so it defaults well below the
# standalone mp3 bitrate: a novel at 192k would run past a gigabyte.
EPUB3_BITRATE = "64k"

# file extensions
TXT_EXT = ".txt"
WAV_EXT = ".wav"
MP3_EXT = ".mp3"
M4B_EXT = ".m4b"
METADATA_FILE = "metadata.json"
CAST_FILE = "characters.json"
SCRIPT_EXT = ".json"
COVER_FILE = "cover.jpg"
SEGMENTS_DIR = "segments"
REJECTED_DIR = "rejected"  # quarantined bad takes + json sidecars for forensics
STATE_FILE = "state.json"

# voice emotions: (instruction, sample_line) for each delivery style
VOICE_EMOTIONS = {
    "neutral": (
        "speaks calmly and clearly",
        "I suppose we should get started then. There is a great deal to discuss.",
    ),
    "happy": (
        "speaks joyfully and warmly, with a smile",
        "This is exactly what I was hoping for! I couldn't have asked for better news.",
    ),
    "sad": (
        "speaks with sorrow and melancholy",
        "I never thought it would end this way. Nothing feels the same anymore.",
    ),
    "angry": (
        "speaks with frustration and intensity",
        "How could you possibly think that was acceptable? I trusted you completely.",
    ),
    "fearful": (
        "speaks with fear and anxiety, voice trembling",
        "Did you hear that? Something is out there.",
    ),
    "surprised": (
        "speaks with astonishment and wonder",
        "Wait, you're saying this has been here the whole time? I can hardly believe it.",
    ),
    "whispering": (
        "whispers softly and secretively",
        "Keep quiet and follow me. We can't let them hear us.",
    ),
    "shouting": (
        "shouts emphatically and loudly",
        "Everyone get back! It's not safe here!",
    ),
    "sarcastic": (
        "speaks with dry irony and sarcasm",
        "Oh wonderful, another brilliant plan that definitely won't fail. "
        "I'm sure this one will work out beautifully.",
    ),
    "excited": (
        "speaks with enthusiasm and high energy",
        "You have to see this! I've never seen anything like it!",
    ),
    "contemplative": (
        "speaks thoughtfully and reflectively, with pauses",
        "Perhaps there's more to this than we first realized. "
        "I keep turning it over in my mind.",
    ),
    "tender": (
        "speaks gently and warmly, with soft affection",
        "You don't have to be afraid anymore. I'm right here.",
    ),
    "stern": (
        "speaks firmly and authoritatively, with gravity",
        "I will not ask again. You will do as I say.",
    ),
    "pleading": (
        "speaks desperately, begging and imploring",
        "Please, you have to believe me. I had no other choice.",
    ),
}
EMOTION_KEYS = list(VOICE_EMOTIONS.keys())

# separator for emotion variant filenames (e.g. CharacterName__happy.wav)
EMOTION_SEP = "__"

DEFAULT_CAST = [
    {
        "name": "Narrator",
        "description": "The book's narrating voice.",
        "voice": (
            "Warm, articulate male voice; mature age; measured slow pace; "
            "authoritative yet compassionate."
        ),
        "audition_line": (
            "The history of the valley wasn't written in books, but in the layers "
            "of sediment resting quietly beneath the river."
        ),
    },
    {
        "name": "Extra Female",
        "description": "An unnamed or minor female character.",
        "voice": "Neutral, casual, female voice, older adult; lower than average pitch.",
        "audition_line": (
            "I really don't think we should be going in there without a map; "
            "honestly, it looks dangerous."
        ),
    },
    {
        "name": "Extra Male",
        "description": "An unnamed or minor male character.",
        "voice": (
            "Gruff, textured baritone voice; older adult; relaxed slow speed; weary but kind."
        ),
        "audition_line": (
            "Just hold the light steady for a minute. I've got to get this wire "
            "connected before the generator fails."
        ),
    },
]
