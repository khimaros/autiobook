"""shared constants and configuration."""

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
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_SPEAKER = "Ryan"
MAX_CHUNK_SIZE = 500  # balance between coherence and decode speed
SAMPLE_RATE = 24000

# llm settings
DEFAULT_LLM_MODEL = "gpt-4o"
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 1.0  # initial delay in seconds, doubles on each retry

# audio processing
PARAGRAPH_PAUSE_MS = 500
CHAPTER_PAUSE_MS = 1000

# mp3 export
DEFAULT_BITRATE = "192k"
UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# file extensions
TXT_EXT = ".txt"
WAV_EXT = ".wav"
MP3_EXT = ".mp3"
METADATA_FILE = "metadata.json"
CAST_FILE = "characters.json"
SCRIPT_EXT = ".json"
COVER_FILE = "cover.jpg"
SEGMENTS_DIR = "segments"
STATE_FILE = "state.json"

# showcase emotions: (instruction, sample_line) for each emotion
SHOWCASE_EMOTIONS = {
    "neutral": (
        "speaks calmly and clearly",
        "I suppose we should get started then.",
    ),
    "happy": (
        "speaks joyfully and warmly, with a smile",
        "This is exactly what I was hoping for!",
    ),
    "sad": (
        "speaks with sorrow and melancholy",
        "I never thought it would end this way.",
    ),
    "angry": (
        "speaks with frustration and intensity",
        "How could you possibly think that was acceptable?",
    ),
    "fearful": (
        "speaks with fear and anxiety, voice trembling",
        "Did you hear that? Something is out there.",
    ),
    "surprised": (
        "speaks with astonishment and wonder",
        "Wait, you're saying this has been here the whole time?",
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
        "Oh wonderful, another brilliant plan that definitely won't fail.",
    ),
    "excited": (
        "speaks with enthusiasm and high energy",
        "You have to see this! I've never seen anything like it!",
    ),
    "contemplative": (
        "speaks thoughtfully and reflectively, with pauses",
        "Perhaps there's more to this than we first realized.",
    ),
}

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
        "description": "Neutral, casual, female voice, young adult; lower than average pitch.",
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
