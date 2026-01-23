# ROADMAP

```
[@] bash end-to-end test script

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
```
