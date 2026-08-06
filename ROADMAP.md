# ROADMAP

```
[x] fix review clobbering the shared audit log: run_review was the only writer that did not load-then-append, so each per-chapter (in fact per-batch) save overwrote audit.json with just its own findings, destroying earlier chapters' results plus cast_merge and revise flag entries. review-authored entries now carry phase="review" and are replaced only for the chapters that invocation owns

[x] fix review rejecting valid llm output when a batch's first segment carries punctuation drift (straight vs typographic apostrophe): _locate_span used an exact substring find where _validate_segments aligns on word tokens, so the segment failed to locate and the span start skipped forward to the next segment that did, cutting the segment's own source out of the window it was then validated against. widened location to fall back on the same token alignment, and pinned the span start at the cursor when text cannot be placed

[x] fix extract duplicating whole documents when a non-content wrapper (section/article) sits between a content tag and its nested content tags (div > section > p); the container branch took get_text() on non-content children, leaking the entire subtree which was then emitted again tag by tag. doubled the extracted text for 45/48 documents in the mercy of gods and 64/79 in children of time, which in turn made review reject valid llm output as hallucinated

[x] fix epub3 read-along highlight trailing one paragraph behind the audio: anchors were computed as positions in the emitted-paragraph list but used to index the content-tag walk, and the two diverge as soon as any content tag emits no text (a wrapper div). every fragment id landed on the preceding element. also drop the subfolder and _readalong suffix: output is now export/<slug>.epub

[x] seed: persist the resolved seed in the workdir (seed.json) so resuming a book idempotently reuses it; random on first run, --seed/AUTIOBOOK_SEED override and are recorded; expose --seed on dramatize and the other workdir commands

[ ] export --epub3: carry the speaker onto injected chunk spans (data-speaker plus a declared aob: prefix) so the read-along records who says what; emotion deliberately excluded as a performance direction rather than a property of the text

[x] export --epub3: wrap each chunk's exact text in an injected span so its <par> uses the recorded clip times instead of a proportional estimate (90.6% of chunks on neuromancer; the rest fall back to paragraph anchoring when the range crosses inline markup). skip a chapter whose extract/ no longer matches the extractor, which silently misplaced every span in it

[x] export --epub3: split a chunk that crosses a paragraph break into one <par> per paragraph, apportioned by character count, instead of anchoring it only where it starts and leaving the later paragraph never highlighted while its audio played
[ ] audition command: add --audition-line flag to override per-character audition lines

[x] export --epub3: emit an epub3 with media overlays (SMIL) so readers highlight text in sync with the narration; granularity follows the audio chunk boundaries already recorded in the timing manifest, coarsened to the narrowest containing element when no per-chunk anchor exists; upgrades epub2 sources to 3.0
[x] config: collapse the three independent 500ms inter-chunk pause definitions (config.PARAGRAPH_PAUSE_MS, pooling.PAUSE_MS_BETWEEN_CHUNKS, audio.concatenate_audio default) into one shared constant; they must agree or every timing in the manifest silently skews

[x] cast generation: let the LLM fold two previously-distinct cast entries into one via a top-level "merges" directive; apply and log to review/audit.json as kind="cast_merge"

[x] review: ignore invalid instruction values from the LLM (keep original rather than silently resetting to "neutral"); log to review/audit.json as kind="invalid_instruction"
[x] export: skip per-chapter export work when state.json shows everything is up to date (no print, no mp3 lyric rewrite); back-fill loop only touches chapters missing the .srt sidecar
[x] review: record per-batch validation rejections to review/audit.json as kind="validation" so the reviewer can see when LLM output was rolled back due to missing/hallucinated text
[x] --accept support in every dramatize phase (cast, audition, emote, script, revise, review, perform, retake, export): re-stamp existing artifacts as fresh under current hashes or skip the phase entirely
[x] dramatize: every phase runs chapter-wise by default (cast→audition→[emote]→script→revise→[review]→perform→retake→export per chapter); --phase-wise restores prior behavior
[x] dramatize: unresolved audit flags no longer block further review passes; gate is enforced only before perform
[x] dramatize: per-chapter mp3 export runs inline in chapter-wise mode so exports land before the next chapter begins
[x] export: embed synchronized lyrics in mp3 id3 tags (SYLT + USLT) so audio-only players render captions without autoloading sidecars
[x] export: emit .srt/.vtt subtitles alongside mp3 in export/ instead of alongside wav in perform/synthesize
[x] review --verbose: list all per-segment changes (speaker/text/instruction) made during review
[x] review: emit only changed segments by index; incremental save; per-batch resume in state.json; progress bar
[x] review: restrict to speaker/instruction corrections (text never modified)
[x] review: LLM human-review flags written to review/flags.json as discovered
[x] flags command: interactive walkthrough / list / clear for review flags
[x] flags: record source_span, show context in walkthrough, [e]dit option that auto-validates
[x] review: record speaker/instruction edits and flags to review/audit.json; `flags` renamed to `audit` (defaults to flags, --all includes edits)
[x] perform + dramatize pipeline: block if unresolved flags exist; bypass with --ignore-flags
[x] perform: fix chapter not reassembling when a segment was regenerated but m_hash matched stored (due to cached wav_sha256 in resume state)
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
