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
DEFAULT_SPEAKER = "ryan"
RETAINED_SPEAKERS = {"Retained", "Unvoiced", "Silent"}  # text kept but not narrated
MAX_CHUNK_SIZE = 500  # balance between coherence and decode speed
SAMPLE_RATE = 24000

# llm settings
DEFAULT_LLM_MODEL = os.getenv("AUTIOBOOK_LLM_MODEL", "openai/gpt-4o")
DEFAULT_THINKING_BUDGET = int(os.getenv("AUTIOBOOK_LLM_THINKING_BUDGET", "16384"))
LLM_TIMEOUT = int(os.getenv("AUTIOBOOK_LLM_TIMEOUT", "600"))
LLM_MAX_RETRIES = 3
VALIDATION_MAX_RETRIES = 5
LLM_RETRY_DELAY = 1.0  # initial delay in seconds, doubles on each retry
CAST_BATCH_SIZE = int(os.getenv("AUTIOBOOK_CAST_BATCH_SIZE", "3"))

# seed for reproducibility (tts + llm). unset → generate one concrete random
# seed per process so the exact value can be logged and recorded with output.
_seed_env = os.getenv("AUTIOBOOK_SEED")
DEFAULT_SEED = int(_seed_env) if _seed_env else random.randint(1, 2**31 - 1)

# tts http settings
TTS_HTTP_TIMEOUT = int(os.getenv("AUTIOBOOK_TTS_TIMEOUT", "300"))

# audio processing
PARAGRAPH_PAUSE_MS = 500
CHAPTER_PAUSE_MS = 1000

# mp3 export
DEFAULT_BITRATE = "192k"
UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# logging
LOG_FILE = "autiobook.log"

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
        "description": (
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
        "description": "Neutral, casual, female voice, older adult; lower than average pitch.",
        "audition_line": (
            "I really don't think we should be going in there without a map; "
            "honestly, it looks dangerous."
        ),
    },
    {
        "name": "Extra Male",
        "description": (
            "Gruff, textured baritone voice; older adult; relaxed slow speed; weary but kind."
        ),
        "audition_line": (
            "Just hold the light steady for a minute. I've got to get this wire "
            "connected before the generator fails."
        ),
    },
]
