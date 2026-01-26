# autiobook

convert epub files to audiobooks using qwen3-tts.

## requirements

- python 3.12+
- ffmpeg
- sox
- uv (python package manager)
- gpu recommended (cuda or rocm)

## installation

```bash
# cuda gpu (default)
make build-cuda

# amd rocm gpu (gfx1151)
make build-rocm

# cpu only
make build-cpu
```

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
autiobook synthesize workdir/ -s Ryan
```

creates:
- `workdir/synthesize/NN_Title.wav` - audio files
- `workdir/synthesize/state.json` - resumability state

### export

convert wav files to mp3 with metadata.

```
autiobook export workdir/ -o audiobook/
```

creates:
- `audiobook/NN_Title.mp3` - mp3 files with id3 tags
- `workdir/export/state.json` - resumability state


### dramatized conversion (llm)

generate a full cast performance using openai-compatible llm
(including llama.cpp) and voice cloning.

```bash
# 1. extract text
autiobook extract book.epub -o workdir/

# 2. generate cast list (using llm)
autiobook cast workdir/ --api-key sk-...

# 3. generate voice auditions (review/edit characters.json first if needed)
autiobook audition workdir/

# 4. create dramatized script (using llm)
autiobook script workdir/ --api-key sk-...

# 5. validate script against source (optional)
autiobook validate workdir/

# 6. fix any issues found (optional)
autiobook fix workdir/ --api-key sk-...

# 7. perform the script (voice cloning)
autiobook perform workdir/

# 8. export to mp3
autiobook export workdir/ -o audiobook/
```

or run the full dramatization pipeline in one go:

```bash
autiobook dramatize workdir/ --api-key sk-...
```

### script validation and repair

after generating scripts, you can validate that all source text is covered
and detect any hallucinated content:

```bash
# check for both missing text and hallucinated segments
autiobook validate workdir/

# check only for missing text
autiobook validate workdir/ --missing

# check only for hallucinated segments
autiobook validate workdir/ --hallucinated
```

to fix issues found during validation:

```bash
# fill missing text and remove hallucinated segments
autiobook fix workdir/ --api-key sk-...

# only fill missing text (uses LLM with surrounding context)
autiobook fix workdir/ --missing --api-key sk-...

# only remove hallucinated segments (no LLM needed)
autiobook fix workdir/ --hallucinated

# control context amount for LLM (characters or paragraphs)
autiobook fix workdir/ --missing --context-chars 1000 --api-key sk-...
autiobook fix workdir/ --missing --context-paragraphs 3 --api-key sk-...
```

### options

- `-o, --output DIR` - output directory
- `-s, --speaker NAME` - tts voice (default: Ryan)
- `-c, --chapters RANGE` - chapter selection (e.g., 1-5, 3,7,10)
- `-v, --verbose` - verbose output

### available voices

Vivian, Ryan, Sunny, Aria, Bella, Nova, Echo, Finn, Atlas

## output

creates one mp3 file per chapter:

```
audiobook/
├── 01_Introduction.mp3
├── 02_Chapter_One.mp3
└── ...
```

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
├── audition/              # character voice samples
│   ├── Character.wav
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
