# architecture

## overview

Standard Workflow:
```
epub file → extract → txt files → synthesize → wav files → export → mp3 files
```

Dramatization Workflow:
```
txt files → cast gen → characters.json → audition → voice samples
     ↓
script gen (llm) → json scripts → validate → fix → perform (cloning) → wav files
```

each phase is idempotent and can be run independently.

## cli commands

### chapters

list chapters in an epub file.

```
autiobook chapters book.epub
```

### extract

extract chapter text from epub to workdir.

```
autiobook extract book.epub -o workdir/
```

creates:
- `workdir/extract/metadata.json` - book metadata
- `workdir/extract/NN_Title.txt` - chapter text files
- `workdir/extract/state.json` - resumability state

### synthesize

convert text files to wav audio.

```
autiobook synthesize workdir/ -s Ryan
```

creates:
- `workdir/synthesize/NN_Title.wav` - audio files
- `workdir/synthesize/state.json` - resumability state

### dramatize / cast / audition / showcase / script / validate / fix / perform

advanced workflow for multi-speaker dramatization.

- `cast`: generates `characters.json` from text sample using LLM.
- `audition`: generates `audition/Character.wav` using `Qwen3-TTS-VoiceDesign`.
- `showcase`: generates `showcase/Character/*.wav` emotion samples using voice cloning.
- `script`: rewrites text into `NN_Title.json` script with speaker attribution using LLM.
- `validate`: verifies scripts match source text, reports missing and hallucinated segments.
- `fix`: fills missing segments using LLM with context, removes hallucinated segments.
- `perform`: synthesizes audio using `Qwen3-TTS-Base` voice cloning from scripts + voice samples.

### showcase

generates emotion samples for each character voice to help vet actor voices in different situations.

```
autiobook showcase workdir/
```

creates:
- `workdir/showcase/CharacterName/emotion.wav` - samples for each emotion
- `workdir/showcase/state.json` - resumability state

emotions generated: neutral, happy, sad, angry, fearful, surprised, whispering, shouting, sarcastic, excited, contemplative.

### validate

compares script segments against original text to detect:
- **missing**: text from source not present in any script segment
- **hallucinated**: script segments with text not found in source

```
autiobook validate workdir/ [--missing] [--hallucinated]
```

### fix

repairs script issues found by validation:
- fills missing text by sending to LLM with surrounding context
- removes hallucinated segments from script
- checkpoints after each fix for resumability

```
autiobook fix workdir/ [--missing] [--hallucinated] --api-key sk-...
autiobook fix workdir/ --missing --context-chars 1000  # character-based context
autiobook fix workdir/ --missing --context-paragraphs 3  # paragraph-based context
```

### export

convert wav files to mp3 with metadata.

```
autiobook export workdir/ -o audiobook/
```

creates:
- `audiobook/NN_Title.mp3` - mp3 files with id3 tags
- `workdir/export/state.json` - resumability state

### convert

run all phases (extract → synthesize → export).

```
autiobook convert book.epub -o workdir/
```

## modules

### epub.py

parses epub files using ebooklib, extracts chapter text using beautifulsoup.

key types:
- `Chapter(index, title, text)` - single chapter data
- `Book(title, author, chapters)` - parsed book data

### tts.py

wraps qwen3-tts for text-to-speech conversion.

- chunks long text at sentence boundaries (~500 char limit)
- synthesizes each chunk and concatenates audio
- supports configurable voice and style
- **Voice Design**: generates new voices from text descriptions
- **Voice Cloning**: clones voices from reference audio

### dramatize.py

orchestrates the dramatization workflow.

- manages cast generation and storage
- handles script generation and parsing
- performs multi-speaker synthesis using `tts.py`

### llm.py

interface for LLM operations (cast and script generation).

- uses openai-compatible API
- provides structured output parsing for cast and scripts

### audio.py

audio processing utilities.

- concatenate audio arrays with pauses
- normalize audio levels

### export.py

mp3 export with id3 metadata.

- wav to mp3 conversion via pydub/ffmpeg
- id3 tags: title, album, artist, track number
- filename format: `NN_Chapter_Title.mp3`

### main.py

cli entry point with subcommands.

## dependencies

| package | purpose |
|---------|---------|
| qwen-tts | text-to-speech |
| openai | llm integration |
| ebooklib | epub parsing |
| beautifulsoup4 | html text extraction |
| pydub | audio manipulation |
| torch | model inference |
| soundfile | wav i/o |

## constants

- `MAX_CHUNK_SIZE = 500` - max chars per tts chunk
- `SAMPLE_RATE = 24000` - qwen3-tts output rate
- `PARAGRAPH_PAUSE_MS = 500` - pause between paragraphs
- `DEFAULT_BITRATE = "192k"` - mp3 encoding bitrate

## workdir structure

Intermediate files are organized into subdirectories by command:

```
workdir/
├── extract/               # extracted text and metadata
│   ├── metadata.json
│   ├── cover.jpg
│   ├── NN_Title.txt
│   └── state.json
├── cast/                  # character list and analysis state
│   ├── characters.json
│   └── state.json
├── audition/              # character voice samples
│   ├── Character.wav
│   └── state.json
├── showcase/              # emotion samples for character voices
│   ├── Character/
│   │   ├── neutral.wav
│   │   ├── happy.wav
│   │   ├── ...
│   │   └── contemplative.wav
│   ├── segments/          # segment cache
│   └── state.json
├── script/                # dramatized scripts (speaker segments)
│   ├── NN_Title.json
│   └── state.json
├── perform/               # dramatized audio performance
│   ├── NN_Title.wav
│   ├── segments/          # segment cache
│   └── state.json
└── synthesize/            # standard mono-voice audio
    ├── NN_Title.wav
    ├── segments/          # segment cache
    └── state.json
```

Each command is fully resumable based on content hashes stored in `state.json`.
