# ROADMAP

```
[x] perform: never spend a tts request on text with nothing to read aloud. mistborn's prologue script carries seven segments whose entire text is a lone left double quote: the script llm emits the resumption quote as its own segment whenever narration interrupts dialogue ("One would think," / " Tresting noted, " / the bare quote / "that a thousand years of working in fields..."). the qwen server answers a bare punctuation mark with a 500, which is in TTS_RETRY_STATUS, so each one burns the full retry budget before failing the run -- the prologue died at segment 0 of 130. a segment with no alphanumeric character never reaches the engine, and never reaches retake either, whose "silent" heuristic would otherwise flag whatever stands in for it and spend five more attempts on the same request. the script phase stops producing them as well: merge_unspeakable_segments folds a wordless segment into a neighbour sharing its speaker, preferring the next one since an opening quote belongs to the line it opens. text is only moved, so the concatenation still matches the source and revise still validates; the split path parses without the merge because its parts must stay as the model drew them

[x] http tts: retry transient transport failures with exponential backoff. a provider that drops the socket raises http.client.RemoteDisconnected straight out of urlopen -- neither an HTTPError nor a RuntimeError, so no handler in the tts path caught it and a 1.5 hour perform run died at segment 180/1963 on "Remote end closed connection without response". the llm path has had backoff since early on; the tts path had none. connection resets, timeouts and 429/5xx replies are retried; a 4xx (bad body, unknown voice) still fails fast because retrying only bills for the same rejection, and a stream that has already handed audio to the player is not replayed. every request now goes through one _request wrapper, which also DRYs the HTTPError-to-RuntimeError conversion that had been spelled out six times, and the two near-identical sse readers collapsed into one -- the buffered one had been joining separately-encoded base64 deltas before decoding, which silently truncates at the first embedded padding

[x] dramatize: make phase-wise the default and add --chapter-wise for the old behaviour; always run cast across every chapter first. the script phase keys its resume hash on the whole cast, so a character discovered in chapter N invalidates the scripts of chapters 1..N-1 -- on mistborn (229 chapters) chapter 4's cast pass sent chapter 1, a table of contents, back through the llm, and each chapter would have invalidated every chapter before it. cast/audition/emote were never chapter-scoped anyway (only run_cast_generation took a chapter list; audition and emote always cover the whole cast), so the head phases now run once up front in both modes and only the tail phases (script/revise/review/perform/retake/export) vary

[x] http tts: recover from a stale server-side clone voice. voice ids from /audio/voices are a sequential in-memory counter on the qwen server, cached per process and never evicted, so a model swap or restart leaves every later batch failing with `unknown voice 'voice_1'` and no way back

[x] cast: accept a merged character's audition line instead of keeping the first one. the line was pinned because changing it invalidates the audition hash, and under the old chapter-wise order a chapter-20 merge would discard voices rendered back at chapter 1. cast now completes across every chapter before audition runs, so the line settles before any audio exists and the churn is free. the voice prompt was already accepted on merge and is the other half of the same hash, so the pin was only ever avoiding one of the two triggers

[x] perform: take ref_text from what the reference wav was rendered with, not the cast's current line. ref_text tells the clone model what the reference audio says, and the audition/emote resume entries record the exact text used; reading the cast instead meant a reworded line (or `audition --accept`, which marks an existing wav fresh under the current cast without re-rendering) could describe words the clip does not speak. not part of the segment hash, so cached takes are unaffected. the perform char_hash drops audition_line for the same reason: the line produces a reference clip and ref_wav_sha already pins the clip's bytes, so keeping it would have discarded every performed segment for a character on a cosmetic reword. extracted as character_hash()

[x] cast: stop asking the llm for an audition line and default every clip to the neutral emotion's sample line, which is what the emote phase already did -- the accent and delivery the character line was meant to carry live in the voice prompt, which is what actually reaches VoiceDesign. removes a field the llm reworded on every merge (each reword costing a re-audition) and a validation rule it regularly failed. audition_line stays on the character as a user-supplied override, set by hand in characters.json or `design --text`, and is dropped on parse so an llm proposal can never clobber it; Character.audition_text() resolves --audition-line > per-character > shared

[x] cast: --llm-audition-lines lets the llm write a per-character audition line, accepted only where none is set. the merge fills a gap and never overwrites, so a line written by hand survives every later cast pass; `design --text` passes overwrite_audition_line to change one deliberately. the flag sits in the shared cast arg group, so `cast` and `dramatize` both get it. off by default: the prompt keeps telling the model not to invent one and the parser keeps dropping it, so nothing changes for runs that do not ask

[ ] cast: refuse pronouns as aliases, in the prompt and deterministically. an alias resolves script speakers, so a pronoun alias silently captures every segment the script attributes to that word: measured across three books, 8 of them were already stored, including Vin owning both "she" and "her" and the Steel Inquisitor owning "it". the prompt names them explicitly now, and a stopword filter drops whole-alias pronoun matches on the way in -- applied to the merged set rather than just the incoming one, so casts that already carry them are cleaned on the next pass. only whole aliases match: "the steward" is a legitimate narrator alias, "she" is not

[x] cast: align the voice prompt with the dimensions Qwen3-TTS VoiceDesign documents reading, and offer its vocabulary. the alibaba voice-design guide (linked from the upstream README) enumerates gender, age, pitch, speed, emotion, timbre and use case with concrete value sets; volume, clarity and fluency appear nowhere, and the guide's own conciseness principle warns against padding with synonyms. measured across 75 audition takes in books/, characters described as quiet averaged -21.5 dBFS rms and those described as loud -20.2 (se 1.17, t=1.11) -- so the 142 voice prompts specifying a volume were spending tokens on an attribute the model does not read. the spec now lists each dimension with the documented values and says to prefer them to invented synonyms, and the worked example was rewritten in that vocabulary since an example steers harder than a rule. accent is kept despite its absence from the list: the model card claims 10 languages plus dialect profiles, and the casts lean on it. FOLLOW-UP: the first cut over-corrected -- "accent where the prose establishes one" plus "omit a dimension rather than padding it out" produced a 1q84 cast with 0/8 accents (a translated novel renders no dialect) and near-identical voices, Komatsu and a walk-on taxi driver differing by one word. accent is now always required, sourced from written dialect where there is one and from the character's origin in the story otherwise; the closed vocabularies are a starting point rather than a ceiling, with a distinguishing phrase invited where they are too coarse; and the cast summary already in context is pointed at with "a cast in which two people share a voice is a failed cast"

[ ] http tts: voice cloning against openrouter via stateless input_references (base64 reference audio plus transcript sent with each request) instead of the qwen server's voice ids; currently refused on the openai dialect in favour of --preset-voices

[x] revise: break the "no segments provided" fix loop. when every segment of a chunk is hallucinated, removing them empties the script, and the empty-script guard then reported the literal placeholder "no segments provided" as a missing fragment. that placeholder was handed to the llm as MISSING TEXT, which converted it into a segment, which validation flagged as hallucinated, which emptied the script again -- a closed 2-cycle that burned all 6 attempts (~20min of local inference) and could never converge. neuromancer chapter 11_EIGHT chunk 2 failed this way three times. validation now reports the real uncovered source text instead of a placeholder, so the fix pass is asked to convert actual book text; removing hallucinations re-validates in the same attempt rather than spending one; and the json repair turn tells the model to redo the original conversion instead of reformatting its own reply (the entry to the loop was a chat preamble, "thank you for sharing this text", that the repair turn dutifully wrapped in the segment schema)

[x] audition --directed: start every character from the configured seed, on both paths. [n]ext and [e]dit roll config.seed to a fresh random value and the pregen worker sets it as a side effect, but nothing put it back, so a character's first take depended on how many times the previous character had been skipped past -- the same cast auditioned twice would not reproduce. the buffered (non-streaming) path was worse: it rolled its own seed for every take and never read config at all, so which seed a character opened on depended on whether streaming was available. the pregen queue now carries a one-shot seed for the take after a flush, and both paths open on the configured seed and roll randomly only for later takes

[x] audition --directed: show the pending cast roster and confirm before starting. a directed session walks every character one at a time and can run for an hour, so a cast with junk entries or duplicates is worth catching at the top rather than character 30. covers both directed paths (voice-design and --preset-voices casting). the roster lists only characters the session would actually stop on -- already-approved voices are not work and are not shown, and an entirely finished cast is not prompted at all. approval is recorded against the cast contents, so resuming an interrupted session goes straight back to work and only an edited cast asks again. DRYs the audition task hash, which had been spelled out at three call sites

[x] llm: stop a content-less reply from killing the run. a model that leaves its whole answer in reasoning_content and returns empty content raised past the validation loop, and retry_with_backoff then re-sent the identical seeded body four times -- deterministic, so four guaranteed repeats of the same failure before the run died mid-chapter (neuromancer 11_EIGHT, twice). the answer is now salvaged from the reasoning field when it parses, and otherwise becomes a feedback turn like any other bad response, since only a changed conversation can change the reply. EmptyResponseError is excluded from backoff: the request went through, so it is not an api error

[x] script: redo the conversion when a chunk validates down to nothing, instead of routing the whole chunk through the fix prompt. the fix prompt is written for filling a gap against surrounding script, and an emptied chunk has neither surrounding script nor a gap smaller than the whole. the redo carries active_seed()+attempt because an identical seeded request only replays the reply that just failed

[x] perform --verbose: name the segment speaker under preset voices. cloning labels each line with the reference wav stem (Alice__angry), but preset mode has no ref wav and printed only the backend voice id (preset:ash), so a verbose log could not be read back against the script

[x] fix audit entries pointing at the wrong segment: entries record a fixed segment number, but revise's mixed-quote splitting inserts segments and renumbers everything after them, so a flag raised before a split drifts (4 of 7 flags on children_of_time chapter 5 were stale, by up to 7 positions). [a]pply wrote the corrected speaker to whatever line now sat at that index -- silent corruption of a different sentence. entries now resolve by their recorded text, falling back to the number only when the text is ambiguous or gone, and the walkthrough says so when the two disagree

[x] audit walkthrough: show the script segments around a flag (the flagged one arrowed, neighbours for context) inline and in full under [p]ager. previously only the flag's recorded text was visible, and only via the pager, so a split decision could not be judged at all -- the whole question is where one segment ends and the next begins

[x] audit walkthrough: [s]uggest asks the review llm to resolve the flag in front of you, re-reviewing that segment with neighbouring context and storing the result as the entry's suggestion so [a]pply lights up. flags mostly arrive without one (revise-declined splits carry only reasoning, review flags what it could not decide), which left the walkthrough with nothing to offer but dismiss. the audit command gained the scripting arg group, having previously had no way to reach an llm at all

[x] audit walkthrough: handle EOF/^C at the prompt as quit instead of raising out of cmd_audit; share one prompt_choice helper with the casting and audition loops, which already did this. detach stdin from spawned audio players so a running take cannot swallow keystrokes meant for the prompt

[x] audit walkthrough: drop [k]eep, which fell through to the same advance as [n]ext -- three keys for two behaviours. the remaining set says what each does to the flag: [a]pply writes the suggestion, [d]ismiss clears the entry so its chapter stops being deferred from perform, [n]ext advances and leaves it open

[x] retake: detect clipping by consecutive run rather than sample count. the old rule flagged any take with >10 samples at/above 0.99 anywhere in it, which fails on a backend that peak-normalizes every clip: measured gemini output puts 13/40 takes at exactly the int16 ceiling with runs of at most 6 samples (0.25ms, inaudible), and the count grows with take length, so the same voice passed at 3s and failed at 12s. across 251 local qwen segments not one sample ever reached 0.99, so the count rule had never once fired on the corpus it was calibrated against. now flags a run of 16+ samples pinned at full scale, which is what flat-topping actually looks like; verified clean on all 291 real segments and still catching a hard-clipped signal

[x] review: refuse to voice retained segments. front-matter blurbs and their attribution lines were generated as "Retained" and then promoted to "Narrator" by review, which then read "-- Financial Times" aloud. three causes, all fixed: the shared rule "use Narrator for ALL unquoted text" outranked the narrower Retained rule and named no front-matter case; the review [Character List] omitted Narrator/Extra/Retained while telling the model to use names exactly as listed, so "Retained" read as a mis-attribution; and nothing enforced the prompt's "do NOT alter Retained segments". such changes are now dropped and logged to the audit as kind="retained_edit", mirroring invalid_instruction

[x] tts: strip leading dashes before synthesis (script text is untouched, so segment hashes and cached takes stay valid). an attribution line's em dash is typographic, and gemini answers the bare dash with silence, which retake burned all 5 attempts on before failing the run

[x] retake: stop reporting seed= on backends that ignore it. hosted providers document no seed on /audio/speech and the openai dialect never sends one, so the logged seed was a value that never left the process; retries there still explore fresh samples, they just are not reproducible. engines now advertise `seeded`

[x] fix --step never advancing past audition with --preset-voices: run_casting rewrote voices.json on every run, including one where every character was already cast and nothing was prompted. --step raises StepComplete when a phase's directory mtime moves, so each re-run stopped on audition again and script was never reached. save_voices is now a no-op when the mapping on disk is identical, matching ResumeManager.update

[x] directed casting: [p]rev to step back to an earlier preset voice (only the non-preset directed audition had it; the casting loop walked the voice list one way with no way back), with a position indicator matching that prompt. going back replays the cached take rather than re-rendering it

[x] directed casting: play preview takes asynchronously so the prompt stays usable while audio plays (it previously blocked on ffplay for the whole take, so every answer waited out the audition line), and stream takes into the player as they render. hosted providers stream their response body rather than sse, so progressive playback now works there too; replay reuses the cached wav rather than paying for the take twice

[x] http tts: fix dramatize --preset-voices rebuilding its audition config from scratch (api_base and a hardcoded qwen model only), which dropped the api key and 401'd every take on a hosted backend; derived from the design config now. --tts-model gained an AUTIOBOOK_TTS_MODEL default and is honoured by the design and clone configs, which previously ignored it and sent qwen model ids

[x] http tts: support hosted openai-compatible backends (openrouter, openai). bearer auth on every tts request (previously llm-only, so any hosted endpoint 401'd), a "dialect" that narrows the request body to the subset they accept and skips the sse probe they would bill twice for, pcm/mp3 response decoding (openrouter never returns wav), preset voice lists for backends with no /audio/voices endpoint (gemini tts names built in), --tts-direction for providers that drop the instructions field, and --tts-api-base/--tts-api-key so tts and llm can point at different providers

[x] fix UnboundLocalError crashing `audition --directed` on a resumed run: quit_requested was bound inside the per-character loop, after the skip `continue`, so a fully-cached cast reached the post-loop check with the name unbound. bound before the loop; the per-character reset was unreachable and removed

[x] cast --verbose: show the voice prompt on new characters, diff every field update (including voice, which was silently dropped when merging into an existing character), report proposed audition_line changes that were kept, name unchanged characters, and summarise each chunk's outcome

[x] cast: split Character.description into a high-level character description and a targeted voice-design prompt, so the text fed to VoiceDesign is purely acoustic instead of half backstory; audition's interactive editor adjusts the voice prompt and is rebound from [d]escribe to [e]dit

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
