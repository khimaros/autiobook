# architecture

## overview

Standard Workflow:
```
epub file → extract → txt files → synthesize → wav files → retake → export → mp3 files
```

Dramatization Workflow:
```
txt files → cast gen → characters.json → audition → emote → voice samples
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

### dramatize / cast / audition / emote / script / revise / perform / callback

dramatize accepts `--strict` as a rollup for `--revise --retake --callback`; convert accepts `--strict` as a rollup for `--retake`.


advanced workflow for multi-speaker dramatization. pipeline order: cast → audition → emote → script → revise → perform → retake.

- `cast`: generates `characters.json` from text sample using LLM. each entry
  carries a `description` (who they are) and a separate `voice` (the
  VoiceDesign prompt). only `voice` reaches the tts model, so backstory prose
  cannot dilute the acoustic traits being asked for; a cast written before the
  split has no `voice` key and falls back to `description`, which keeps its
  generated voices and cached segments identical.
- `audition`: generates `audition/Character.wav` using `Qwen3-TTS-VoiceDesign` with the character description only (no emotion hints). this is the canonical per-character voice identity and serves as a fallback ref clip during perform when an emotion variant is missing. `--callback` validates inline.
- `emote`: generates `emote/Character__emotion.wav` per emotion using `Qwen3-TTS-VoiceDesign` with the description plus an emotion instruct. these are the per-emotion ref clips that perform clones from. reuses the per-character seed recorded by audition so every variant rides the same voice trajectory; a changed audition seed invalidates the emote via the task hash. `--callback` validates inline.
- `callback`: post-hoc audio quality scan for `audition/` and `emote/` wavs (base files and per-emotion variants); deletes offenders and regenerates (mirrors `retake` for chapter segments). `--dry-run` reports only; `--prune` deletes without regenerating.
- `script`: rewrites text into `NN_Title.json` script with speaker attribution using LLM. Supports `--validate` for iterative fixing of missing or hallucinated segments during generation.
- `revise`: review and repair scripts. compares script to source, then fills missing segments via LLM and removes hallucinated segments. `--dry-run` reports without modifying; `--prune` strips hallucinations but skips LLM fix-missing.
- `perform`: synthesizes audio using `Qwen3-TTS-Base` voice cloning from scripts + voice samples.

### review

corrections the review LLM emits are filtered before they land. an instruction
outside `EMOTION_KEYS` is dropped, and so is any change that moves a segment
out of `RETAINED_SPEAKERS` -- retained text is the material that must never be
spoken (section markers, chapter numbers, front-matter blurbs and their
attribution lines), and promoting it to Narrator is never a correction. both
are recorded to `review/audit.json` (`kind="invalid_instruction"` and
`kind="retained_edit"`) rather than applied silently.

the prompt asks for the same restraint, but a rule is not a guarantee: the
shared script rules also say to narrate all unquoted text, and that broader
instruction is what pulled blurbs out of Retained in the first place.

### audition / emote

the interactive audition prompt offers `[e]dit`, which opens `$EDITOR` on the
voice prompt and re-synthesizes; accepting a take writes the revised prompt
back to `characters.json`.

`audition` produces one base file per character (`audition/{name}.wav`) using `design_voice` with the character description, tracked in `audition/state.json`. `emote` then produces per-emotion variants (`emote/{name}__{emotion}.wav`) using `design_voice` with the description plus an emotion instruction, tracked in `emote/state.json` keyed `{name}/{emotion}`. emote reads the seed recorded in `audition/state.json` for each character and reuses it, so the base file and all emotion variants stay on the same voice trajectory; that seed also feeds the emote task hash, so bumping an audition seed forces re-emote. both phases honor `--callback` and archive rejected takes to `{phase}/rejected/`.

```
autiobook audition workdir/
autiobook emote workdir/
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

`--epub3` instead rebuilds the source epub as a read-along with media
overlays, so readers highlight text in sync with the narration:

```
autiobook export workdir/ --epub3
```

creates `workdir/export/<slug>.epub`.

### convert

run all phases (extract → synthesize → retake → export).

```
autiobook convert book.epub -o workdir/
```

## modules

### epub.py

parses epub files using ebooklib, extracts chapter text using beautifulsoup.

key types:
- `Chapter(index, title, text, href)` - single chapter data
- `Book(title, author, chapters)` - parsed book data

`extract_paragraphs_from_html` returns `(tag_index, text)` per paragraph;
the index is the position within the content-tag walk and is what lets
`overlay.py` map a span of extracted text back to its source element.

### tts.py

wraps qwen3-tts for text-to-speech conversion.

- chunks long text at sentence boundaries (~500 char limit)
- synthesizes each chunk and concatenates audio
- supports configurable voice and style
- **Voice Design**: generates new voices from text descriptions
- **Voice Cloning**: clones voices from reference audio

### tts_http.py

talks to a speech api over http instead of loading a model locally.

two dialects share one client, resolved from the endpoint host unless pinned
with `--tts-dialect`:

- `qwen` -- the local qwen3-tts server. its api is a superset of openai's:
  wav responses, sse streaming with usage/timings, sampler fields, and an
  `/audio/voices` endpoint for both listing presets and minting cloned voices
  from reference audio.
- `openai` -- hosted providers (openrouter, openai). requests carry only the
  documented fields, responses are raw pcm, and the sse probe is skipped
  because a metered endpoint bills it as a second synthesis of the same text.

the split exists because the two disagree on every axis that matters: a body
field the local server needs is at best ignored and at worst a paid 400 on a
hosted one, and `wav` is not among the formats openrouter will return.

capabilities that need server-side state (voice design, cloning) are refused
on the openai dialect rather than silently producing one voice for the whole
cast; `--preset-voices` is the supported route there. voice discovery has no
hosted equivalent at all, so preset names come from `--tts-voices` or a
built-in per-model table.

`--tts-direction` picks the channel for delivery direction: a top-level
`instructions` field, or folded into the input text for providers that drop
unknown fields.

engines advertise two capabilities the pipeline branches on. `seeded` is false
for hosted providers, which document no seed on `/audio/speech`: retakes there
still explore fresh samples, so retrying is worthwhile, but the seed is not
reported since it never leaves the process. text is normalized at the single
synthesis chokepoint (`_run_synthesis`) rather than at task construction, so
segment hashes stay keyed on the script text and cached takes survive.

progressive playback works on both, by different means: the qwen server emits
pcm inside sse deltas once given a batch size, while a hosted provider's
response body is itself the stream and is read a buffer at a time. the
`streaming` property hides which, so the interactive loops ask the engine
rather than inspecting config.

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

### overlay.py

epub3 media overlay (read-along) export.

- anchors each timing-manifest chunk to a source element via the same
  token alignment `dramatize.py` uses to validate scripts
- merges chunks sharing an element into one `<par>`, since a `<par>` binds
  exactly one text fragment to one audio clip
- clip ranges extend to meet the next `<par>` so inter-chunk pauses play
  rather than being skipped
- copies the source epub entry-for-entry, touching only the narrated
  documents, the package document, and the new smil/audio files
- upgrades epub2 sources to 3.0, deriving a nav document from the ncx

granularity follows the audio chunk boundaries: each chunk's exact text is
wrapped in an injected `<span>` so its `<par>` carries the recorded clip
times rather than an estimate. on a representative novel 90.6% of chunks
place this way; the rest fall back to the containing paragraph, apportioned
by character count, when the range crosses inline markup or a paragraph
break. injection is refused unless the element's collapsed text matches the
paragraph the offsets were computed against, and a chapter whose `extract/`
no longer matches the extractor is skipped outright -- both mismatches would
otherwise place every span in the chapter at an arbitrary offset.

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
- `PARAGRAPH_PAUSE_MS = 500` - pause between chunks; chapter assembly and the
  timing manifest must agree on this or every cue and overlay offset drifts
- `DEFAULT_BITRATE = "192k"` - mp3 encoding bitrate
- `EPUB3_BITRATE = "64k"` - audio embedded in a read-along epub3
- `SEED_FILE = "seed.json"` - per-book seed, resolved once and reused

the seed is resolved in `main()` once the workdir is known, then pinned with
`config.set_active_seed`. tts and llm read it lazily (`field(default_factory=
active_seed)` and a `None` sentinel) because module-level defaults would bind
at import, long before the command line is parsed.

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
