# architecture

## overview

Standard Workflow:
```
epub file → extract → txt files → synthesize → wav files → retake → export → mp3 files
```

Dramatization Workflow:
```
txt files → cast gen → characters.json → audition → voice samples
     ↓
script gen (llm) → json scripts → revise → perform (cloning) → wav files → retake
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
autiobook synthesize workdir/ -s ryan
```

creates:
- `workdir/synthesize/NN_Title.wav` - audio files
- `workdir/synthesize/state.json` - resumability state

### dramatize / cast / introduce / audition / script / revise / perform / callback

dramatize accepts `--strict` as a rollup for `--revise --retake --callback`; convert accepts `--strict` as a rollup for `--retake`.


advanced workflow for multi-speaker dramatization. pipeline order: cast → introduce → audition → script → revise → perform → retake.

- `cast`: generates `characters.json` from text sample using LLM.
- `introduce`: generates `introduce/Character.wav` using `Qwen3-TTS-VoiceDesign` with the character description only (no emotion hints). this is the canonical per-character voice identity and serves as a fallback ref clip during perform when an emotion variant is missing. `--callback` validates inline.
- `audition`: generates `audition/Character__emotion.wav` per emotion using `Qwen3-TTS-VoiceDesign` with the description plus an emotion instruct. these are the per-emotion ref clips that perform clones from. reuses the per-character seed recorded by introduce so every variant rides the same voice trajectory; a changed introduce seed invalidates the audition via the task hash. `--callback` validates inline.
- `callback`: post-hoc audio quality scan for `introduce/` and `audition/` wavs (base files and per-emotion variants); deletes offenders and regenerates (mirrors `retake` for chapter segments). `--dry-run` reports only; `--prune` deletes without regenerating.
- `script`: rewrites text into `NN_Title.json` script with speaker attribution using LLM. Supports `--validate` for iterative fixing of missing or hallucinated segments during generation.
- `revise`: review and repair scripts. compares script to source, then fills missing segments via LLM and removes hallucinated segments. `--dry-run` reports without modifying; `--prune` strips hallucinations but skips LLM fix-missing.
- `perform`: synthesizes audio using `Qwen3-TTS-Base` voice cloning from scripts + voice samples.

### introduce / audition

`introduce` produces one base file per character (`introduce/{name}.wav`) using `design_voice` with the character description, tracked in `introduce/state.json`. `audition` then produces per-emotion variants (`audition/{name}__{emotion}.wav`) using `design_voice` with the description plus an emotion instruction, tracked in `audition/state.json` keyed `{name}/{emotion}`. audition reads the seed recorded in `introduce/state.json` for each character and reuses it, so the base file and all emotion variants stay on the same voice trajectory; that seed also feeds the audition task hash, so bumping an introduce seed forces re-audition. both phases honor `--callback` and archive rejected takes to `{phase}/rejected/`.

```
autiobook introduce workdir/
autiobook audition workdir/
```

emotions generated: neutral, happy, sad, angry, fearful, surprised, whispering, shouting, sarcastic, excited, contemplative.

### revise

compares script segments against original text and repairs defects:
- **missing**: text from source not present in any script segment → filled via LLM with surrounding context
- **hallucinated**: script segments with text not found in source → removed
- checkpoints after each revise step for resumability
- `--dry-run`: report only (exits non-zero if issues found); no changes written
- `--prune`: strip hallucinations only; skip the LLM fix-missing pass

```
autiobook revise workdir/ --api-key sk-...
autiobook revise workdir/ --dry-run                    # review only
autiobook revise workdir/ --prune                      # local cleanup only
```

### export

convert wav files to mp3 with metadata.

```
autiobook export workdir/
```

creates:
- `workdir/export/NN_Title.mp3` - mp3 files with id3 tags
- `workdir/export/state.json` - resumability state

### convert

run all phases (extract → synthesize → retake → export).

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
├── introduce/             # per-character base voices (description only)
│   ├── Character.wav
│   └── state.json
├── audition/              # per-emotion voice variants
│   ├── Character__neutral.wav
│   ├── Character__happy.wav
│   ├── ...
│   └── state.json
├── script/                # dramatized scripts (speaker segments)
│   ├── NN_Title.json
│   └── state.json
├── perform/               # dramatized audio performance
│   ├── NN_Title.wav
│   ├── segments/          # segment cache
│   └── state.json
├── synthesize/            # standard mono-voice audio
│   ├── NN_Title.wav
│   ├── segments/          # segment cache
│   └── state.json
└── export/                # final mp3 output
    ├── NN_Title.mp3
    └── state.json
```

Each command is fully resumable based on content hashes stored in `state.json`.
