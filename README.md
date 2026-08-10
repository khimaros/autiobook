# autiobook

convert epub files to audiobooks using qwen3-tts or with any
openai compatible text-to-speech endpoint.

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

### read-along epub3

rebuild the source epub with media overlays, so a reader highlights each
phrase as it is narrated.

```
autiobook export workdir/ --epub3
```

creates `workdir/export/<book>.epub`.

the source epub is located automatically from the extract state; pass
`--epub book.epub` if it has since moved. `--epub3-bitrate` controls the
embedded audio (default `64k` - a novel runs a few hundred MiB, so raising
this gets large quickly).

epub2 sources are upgraded to epub3, which media overlays require.

read-along playback is supported by apple books, thorium, and other
readium-based readers. kindle ignores media overlays entirely, and support
in google play books and kobo is patchy.


### dramatized conversion (llm)

generate a full cast performance using an openai-compatible llm
(including llama.cpp) and voice cloning.

```bash
# 1. extract text
autiobook extract book.epub -o workdir/

# 2. generate cast list (using llm)
#    each character gets a description (who they are) and a separate
#    voice prompt; only the voice prompt is sent to the tts model.
#    --llm-audition-lines also asks for a per-character audition line,
#    filling only characters that don't already have one
autiobook cast workdir/ --api-key sk-...

# 3. generate base voice per character (review/edit characters.json first if needed)
#    every character auditions on the same sentence, so takes differ only in
#    voice. set "audition_line" on a character in characters.json to give that
#    one its own line; --audition-line changes it for the whole run
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

# finish each chapter end to end before starting the next
autiobook dramatize book.epub --chapter-wise
```

each phase runs across all chapters before the next phase starts. `--chapter-wise`
instead completes script → revise → review → perform → retake → export for
chapter 1 before chapter 2 begins, so you hear finished audio sooner. either way
cast, audition and emote run across every chapter first: the script phase keys
its resume state on the whole cast, so casting chapter by chapter would send
every earlier chapter back through the LLM each time a new character turned up.

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

### audit

`audit` walks the flags `review` and `revise` raised for human attention. an
open flag defers its chapter from `perform`, so this is where a stalled
chapter gets unblocked.

```bash
autiobook audit workdir/
autiobook audit workdir/ --list      # non-interactive
autiobook audit workdir/ --all       # include applied-edit records
```

per entry: `[a]pply` writes the suggested correction, `[s]uggest` asks the llm
for one when the flag arrived without it, `[e]dit` opens the chapter script,
`[d]ismiss` clears the flag, `[n]ext` leaves it open for later.

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
- `--tts-api-base`, `--tts-api-key` - point tts at a different provider than the llm
- `--tts-dialect` - request subset: `qwen`, `openai`, or `auto` (by host)
- `--tts-voices` - preset voices for a backend with no `/audio/voices` endpoint
- `--tts-direction` - `field` (top-level `instructions`) or `prefix` (folded into the text)
- `--llm-model` - llm model name
- `--m4b` - export as a single m4b with chapter markers
- `-v, --verbose` - verbose output
- `-f, --force` - ignore resume state
- `--seed` - seed for tts and llm (`0` disables seeding)

### seeds

the seed for a book is chosen once and recorded in `workdir/seed.json`, so
resuming a conversion reuses it instead of drifting to a new one each run.

precedence is `--seed`, then `$AUTIOBOOK_SEED`, then the recorded seed, then a
fresh random one. passing `--seed` (or setting the env var) replaces what was
recorded, and says so, since the whole point is that a book keeps one seed.

environment variables (also loadable from `.env`; see `.env.example`):
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `AUTIOBOOK_LLM_MODEL`,
`AUTIOBOOK_TTS_CLONE_MODEL`, `AUTIOBOOK_TTS_INSTRUCT_MODEL`,
`AUTIOBOOK_TTS_DESIGN_MODEL`, `AUTIOBOOK_SEED`,
`AUTIOBOOK_LLM_THINKING_BUDGET`, `AUTIOBOOK_CAST_BATCH_SIZE`,
`AUTIOBOOK_TTS_MODEL`, `AUTIOBOOK_TTS_API_BASE`, `AUTIOBOOK_TTS_API_KEY`,
`AUTIOBOOK_TTS_DIALECT`, `AUTIOBOOK_TTS_VOICES`, `AUTIOBOOK_TTS_DIRECTION`,
`AUTIOBOOK_TTS_RESPONSE_FORMAT`, `AUTIOBOOK_TTS_MAX_RETRIES`,
`AUTIOBOOK_TTS_RETRY_DELAY`.

### hosted tts backends

the default tts backend is a local qwen3-tts server, whose api is a superset
of openai's: wav responses, sse streaming, and an `/audio/voices` endpoint for
listing presets and creating cloned voices. hosted providers offer none of
that, so `--tts-dialect openai` (selected automatically for `openrouter.ai`
and `api.openai.com`) narrows each request to `model`, `input`, `voice`,
`response_format` and `instructions`, asks for pcm instead of wav, and skips
the sse probe -- on a metered api that probe is a second billed synthesis of
the same text.

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-or-...
autiobook dramatize book.epub \
    --tts-model google/gemini-3.1-flash-tts-preview \
    --preset-voices --tts-direction prefix
```

`--tts-model` (or `$AUTIOBOOK_TTS_MODEL`) sets the model for every mode. the
three per-mode defaults are qwen model ids, so a hosted run that leaves them
in place asks openrouter for a model it has never heard of.

against any http backend, a dropped connection, a timeout or a 429/5xx reply
is retried with exponential backoff, so a blip an hour into a synthesis run no
longer ends it: `AUTIOBOOK_TTS_MAX_RETRIES` (4) retries at
`AUTIOBOOK_TTS_RETRY_DELAY` (2 seconds), doubling each time. every other 4xx
-- a malformed body, an unknown voice, a rejected key -- fails at once, since
the reply will not change and hosted providers bill for each send.

takes play while they render on a hosted backend without any extra setting --
`AUTIOBOOK_TTS_STREAM_BATCH_SIZE` is a qwen-server knob, and the provider's
response body is already a stream. in `--directed` casting the prompt stays
live during playback, so `n` moves on mid-take instead of after it, and
`[p]rev` steps back to an earlier voice (replayed from cache, not re-rendered).

`--directed` opens with the cast roster (names, aliases, descriptions) and waits
for approval before the first take, so a cast with junk entries or duplicates
can be fixed before an hour of takes rather than after. only characters the
session will stop on are listed; approved voices are left off, a finished cast
is not prompted at all, and resuming asks again only if the cast changed. each
character starts from the configured seed, so `[n]ext` on one character does not
change what the next character's first take sounds like, and a character opens
on the same take whether or not the backend streams.

`--preset-voices` is required: hosted providers have fixed voices and no
server-side voice creation, so voice design and cloning are unavailable there.
openrouter publishes no voice discovery api either; the gemini tts voice names
are built in, and anything else needs `--tts-voices Zephyr,Puck,Kore`.

`--tts-direction prefix` folds emotion and voice direction into the input text
rather than the `instructions` field, which openrouter does not document as a
top-level parameter. without it, delivery direction may be silently dropped
and every line comes out flat.

to keep tts local while the llm runs hosted (or the reverse), set
`--tts-api-base`/`--tts-api-key` alongside `--api-base`/`--api-key`.

### available voices

qwen3-tts: Vivian, Ryan, Sunny, Aria, Bella, Nova, Echo, Finn, Atlas

gemini tts (built in, used with `--preset-voices`): Zephyr, Puck, Charon,
Kore, Fenrir, Leda, Orus, Aoede, and 22 more.

## output

creates one mp3 file per chapter in `workdir/export/` (or a single `.m4b`
with chapter markers when `--m4b` is passed):

```
workdir/export/
├── 01_Introduction.mp3
├── 02_Chapter_One.mp3
└── ...
```

The `export/` folder also contains `.srt` and `.vtt` subtitles alongside
each chapter mp3 (with speaker labels for dramatized output).

`--epub3` instead produces a single read-along epub with the narration
embedded and synchronized to the text.

compatible with the [Voice](https://github.com/PaulWoitaschek/Voice) audiobook player for android.

## workdir structure

Intermediate files are organized into subdirectories by command:

```
workdir/
├── seed.json              # seed recorded for this book, reused on resume
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
│   ├── segments/          # segment cache
│   └── state.json
├── synthesize/            # standard mono-voice audio
│   ├── NN_Title.wav
│   ├── NN_Title.wav.timing.json
│   ├── segments/          # segment cache
│   └── state.json
└── export/                # final mp3 output
    ├── NN_Title.mp3
    ├── NN_Title.srt       # subtitles (with speaker labels)
    ├── NN_Title.vtt       # webvtt subtitles
    └── state.json
```

Each command is fully resumable based on content hashes stored in `state.json`.
