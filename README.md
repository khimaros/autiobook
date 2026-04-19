# autiobook

convert epub files to audiobooks using qwen3-tts.

## requirements

- python 3.12+
- ffmpeg
- sox
- uv (python package manager)
- gpu recommended (cuda or rocm) if running tts locally
- or an openai-compatible tts endpoint (no local gpu required)

## installation

```bash
# cuda gpu (default, includes local tts)
make build-cuda

# amd rocm gpu (gfx1151, includes local tts)
make build-rocm

# cpu only (includes local tts)
make build-cpu
```

local tts extras are optional. to drive an openai-compatible tts endpoint
instead, install without the `[local]` extra and set `--api-base` / `OPENAI_BASE_URL`.

## usage

### enter the venv

```bash
source .venv/bin/activate

autiobook --help
```

### list chapters

```bash
autiobook chapters book.epub
```

### full conversion (idempotent)

```bash
autiobook convert book.epub -o workdir/
```

runs all phases, skipping already-completed steps.

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

### export

convert wav files to mp3 with metadata.

```
autiobook export workdir/
```

creates:
- `workdir/export/NN_Title.mp3` - mp3 files with id3 tags
- `workdir/export/state.json` - resumability state


### dramatized conversion (llm)

generate a full cast performance using an openai-compatible llm
(including llama.cpp) and voice cloning.

```bash
# 1. extract text
autiobook extract book.epub -o workdir/

# 2. generate cast list (using llm)
autiobook cast workdir/ --api-key sk-...

# 3. generate base voice per character (review/edit characters.json first if needed)
autiobook audition workdir/

# 4. generate per-emotion voice variants
autiobook emote workdir/

# 5. create dramatized script (using llm)
autiobook script workdir/ --api-key sk-...

# 6. review and repair script (optional; --dry-run to only report)
autiobook revise workdir/ --api-key sk-...

# 7. perform the script (voice cloning)
autiobook perform workdir/

# 8. export to mp3
autiobook export workdir/
```

or run the full dramatization pipeline in one go:

```bash
autiobook dramatize book.epub --api-key sk-...

# pause after each phase for examination
autiobook dramatize book.epub --step

# re-run the last completed phase
autiobook dramatize book.epub --redo

# enable all inline quality checks (script revise + voice/segment retake)
autiobook dramatize book.epub --strict
```

### script revision

after generating scripts, `revise` reviews them against the source text,
detecting missing or hallucinated segments and repairing them via llm:

```bash
# review and repair: fill missing text, remove hallucinated segments
autiobook revise workdir/ --api-key sk-...

# only review; don't modify scripts
autiobook revise workdir/ --dry-run

# local cleanup only: strip hallucinations, skip the llm fix-missing pass
autiobook revise workdir/ --prune
```

### voice and segment quality checks

- `callback` scans `audition/` and `emote/` wavs for silent/clipped/noisy
  takes and re-generates them with a bumped seed.
- `retake` does the same for `perform/` and `synthesize/` segments.
- `locate` looks up which segment wav backs a given audio time position
  (useful for debugging a glitch you heard in the output).

```bash
autiobook callback workdir/
autiobook retake workdir/
autiobook locate workdir/perform/NN_Title.wav 00:12:34
```

### options

- `-o, --output DIR` - workdir for intermediate files (default: `<epub>_output/`)
- `-s, --voice NAME` - tts voice for `synthesize` (default: ryan)
- `-c, --chapters RANGE` - chapter selection (e.g., 1-5, 3,7,10)
- `--tts-model`, `--tts-design-model`, `--tts-clone-model` - override tts models
- `--api-base`, `--api-key` - openai-compatible endpoint (defaults to `$OPENAI_BASE_URL` / `$OPENAI_API_KEY`)
- `--llm-model` - llm model name
- `--m4b` - export as a single m4b with chapter markers
- `-v, --verbose` - verbose output
- `-f, --force` - ignore resume state

environment variables (also loadable from `.env`; see `.env.example`):
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AUTIOBOOK_LLM_MODEL`,
`AUTIOBOOK_TTS_CLONE_MODEL`, `AUTIOBOOK_TTS_INSTRUCT_MODEL`,
`AUTIOBOOK_TTS_DESIGN_MODEL`, `AUTIOBOOK_SEED`,
`AUTIOBOOK_LLM_THINKING_BUDGET`, `AUTIOBOOK_CAST_BATCH_SIZE`.

### available voices

Vivian, Ryan, Sunny, Aria, Bella, Nova, Echo, Finn, Atlas

## output

creates one mp3 file per chapter in `workdir/export/` (or a single `.m4b`
with chapter markers when `--m4b` is passed):

```
workdir/export/
├── 01_Introduction.mp3
├── 02_Chapter_One.mp3
└── ...
```

`perform/` and `synthesize/` also emit `.srt` and `.vtt` subtitles alongside
each chapter wav (with speaker labels for dramatized output).

compatible with the [Voice](https://github.com/PaulWoitaschek/Voice) audiobook player for android.

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
├── audition/              # per-character base voices (description only)
│   ├── Character.wav
│   └── state.json
├── emote/                 # per-emotion voice variants
│   ├── Character__neutral.wav
│   ├── Character__happy.wav
│   ├── ...
│   └── state.json
├── script/                # dramatized scripts (speaker segments)
│   ├── NN_Title.json
│   └── state.json
├── perform/               # dramatized audio performance
│   ├── NN_Title.wav
│   ├── NN_Title.wav.timing.json  # per-chunk start/end offsets + metadata
│   ├── NN_Title.srt       # subtitles (with speaker labels)
│   ├── NN_Title.vtt       # webvtt subtitles
│   ├── segments/          # segment cache
│   └── state.json
├── synthesize/            # standard mono-voice audio
│   ├── NN_Title.wav
│   ├── NN_Title.wav.timing.json
│   ├── NN_Title.srt
│   ├── NN_Title.vtt
│   ├── segments/          # segment cache
│   └── state.json
└── export/                # final mp3 output
    ├── NN_Title.mp3
    └── state.json
```

Each command is fully resumable based on content hashes stored in `state.json`.
