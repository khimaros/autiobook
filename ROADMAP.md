# ROADMAP

```
[ ] audition command: add --audition-line flag to override per-character audition lines

[x] rename phases: audition→emote (per-emotion variants) and introduce→audition (per-character base); emote reuses audition seed
[x] fix_missing_segment: prevent LLM from grabbing text from context instead of MISSING TEXT
[x] fix nested HTML tag duplication in extract_text_from_html (div containing p tags)
[x] make temperature optional for tts/llm (no default; only send when explicitly set)
[x] chapter-ordered segment scheduling with early assembly
[x] merge validate + fix commands into `revise` (with --dry-run for report-only)
[x] retake command + dramatize/convert --retake: detect corrupted segment wavs (silent/click/truncated/clipping/noisy) and regenerate with seed bump
[x] split voice generation into introduce (design_voice description-only base → audition/{name}.wav) and audition (per-emotion design_voice → audition/{name}__{emotion}.wav); remove showcase command; pipeline order cast → introduce → audition → script → ...; both honor --callback and audition/rejected/
[x] split introduce into its own folder (introduce/{name}.wav + introduce/state.json); audition reuses the per-character seed recorded by introduce and folds it into the audition task hash
[x] audition/showcase --callback + `callback` subcommand: audio quality checks for voice samples (mirrors retake for segments); dramatize --callback/--strict, convert --strict rollups; showcase auto-runs callback scan unless --no-callback
[x] script generation: surface short canonical speaker names to LLM (reduce token bloat, improve instruction adherence)
[x] perform --verbose: print voice name and performance line for each segment as it synthesizes
[x] cast generation: remove per-chapter 2000-char truncation so late-introduced character names are captured
[x] dramatize pipeline: run audition phase before script generation to match design diagram
[x] emit .srt and .vtt subtitles alongside chapter wavs during assembly (synthesize + dramatize, with speaker labels)
[x] stronger cast alias prompt: require all prose variants; richer example; nudge alias updates on batch 2+
[x] permissive speaker resolution: auto-fix punctuation/case/unambiguous shortforms
[x] local auto-fix for invalid instructions (reset to neutral, no llm round-trip)
[x] grouped validation feedback with cast list hint
[x] log llm reasoning tokens (reasoning_content / <think> blocks) for retry diagnosis
[x] cast generation: handle dict responses from json_object mode (wrapper keys, single-char, name-keyed)
[x] configurable seed for tts and llm (AUTIOBOOK_SEED env, default 31337, <=0 disables)
[x] change default audiobook output folder from 'export/'
[x] http tts engine: add openai-compatible http backend, make local tts extras optional
[x] script --validate: show detailed info (missing fragment, context, neighboring segments)
[x] logging: autiobook.log with full LLM queries/responses and validation details
[x] script: support "Retained" speaker for section markers, chapter numbers, etc.
[x] script command: validate chunks during generation and retry with feedback on failure
[x] iterative script validation fixing (only re-generate missing/hallucinated segments)
[x] all commands: exit with non-zero code on failure
[x] replace litellm with direct urllib HTTP requests to openai-compatible API
[x] separate TTS model flags: --tts-model, --tts-design-model, --tts-clone-model
[x] --step flag: pause after each pipeline phase for examination
[x] --redo flag: re-run the last completed pipeline phase
[x] hide local-only CLI flags when [local] extra not installed
[x] m4b export support with chapter markers
[x] refactor dramatize/convert commands to DRY flags and improve output directory inference
[x] improve LLM prompts for smaller models
[x] global content addressable store for audio clips to avoid re-generating identical phrases
[x] DRY tts code between audition, perform, and synthesize
[x] hash all voice descriptions and performance segments and use that for save/resume
[x] add "fix" phase to dramatize flow to remove hallucinations and fix missing
[x] add "validate" phase to dramatize flow to verify scripts match source text
[x] dramatize command: pass through TTS flags (--pooled, --batch-size, etc.) and resume
[x] script LLM prompt: narrator handles "X said" portions, characters only voice quoted content
[x] keep chunks unless explicitly cleaned up (add "clean" command)
[x] track character appearances in cast.json, add --min-appearances flag for audition/script
[x] DRY synthesize and perform code (use `concatenate_audio`, add `iter_pending_chapters`)
[x] add character alias tracking to cast command
[x] extract epub cover and embed in mp3 files
[x] granular progress logging (sample/s)
[x] epub parser module
[x] tts engine wrapper
[x] audio processing module
[x] mp3 export module
[x] cli interface
[x] project setup and documentation
[x] improve pyproject.toml for hardware-specific dependencies
[x] make script command idempotent with incremental JSON saves
[x] add LLM retry logic for API errors and invalid JSON
[x] DRY common LLM flags (api-base, api-key, model) into utils.py
[x] DRY command line flag parsing and chapter selection logic
```
