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

### phased conversion

```bash
# extract text from epub
autiobook extract book.epub -o workdir/

# synthesize audio from text (skips existing wav files)
autiobook synthesize workdir/ -s Ryan

# export wav files to mp3 (skips existing mp3 files)
autiobook export workdir/ -o audiobook/
```

### dramatized conversion (llm)

generate a full cast performance using openai-compatible llm
(including llama.cpp) and voice cloning.

```bash
# 1. extract text
autiobook extract book.epub -o workdir/

# 2. generate cast list (using llm)
autiobook cast workdir/ --api-key sk-...

# 3. generate voice auditions (review/edit cast.json first if needed)
autiobook audition workdir/

# 4. create dramatized script (using llm)
autiobook script workdir/ --api-key sk-...

# 5. perform the script (voice cloning)
autiobook perform workdir/

# 6. export to mp3
autiobook export workdir/ -o audiobook/
```

or run the full dramatization pipeline in one go:

```bash
autiobook dramatize workdir/ --api-key sk-...
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

```
workdir/
├── metadata.json          # book metadata
├── 01_Introduction.txt    # extracted chapter text
├── 01_Introduction.wav    # synthesized audio
├── 02_Chapter_One.txt
├── 02_Chapter_One.wav
└── ...
```
