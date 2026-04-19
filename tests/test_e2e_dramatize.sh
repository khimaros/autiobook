#!/bin/bash
# end-to-end test for autiobook
set -euo pipefail

TEST_EPUB="./testdata/isaac-asimov_short-science-fiction_advanced.epub"
WORKDIR="./testdata_workdir/isaac-asimov_short-science-fiction/"
OUTDIR="$WORKDIR/export/"

cleanup() {
	rm -rf "$WORKDIR"
}

die() {
	echo "FAIL: $1" >&2
	exit 1
}

echo "=== autiobook e2e test ==="

. .venv/bin/activate

# cleanup from previous runs
#cleanup

# check test file exists
[[ -f "$TEST_EPUB" ]] || die "test epub not found: $TEST_EPUB"

echo ""
echo "--- test: chapters ---"
autiobook chapters "$TEST_EPUB" || die "chapters command failed"

echo ""
echo "--- test: extract ---"
autiobook extract "$TEST_EPUB" -o "$WORKDIR" || die "extract command failed"

# verify extraction output
[[ -f "$WORKDIR/extract/metadata.json" ]] || die "metadata.json not created"
txt_count=$(find "$WORKDIR/extract" -name "*.txt" | wc -l)
[[ $txt_count -gt 0 ]] || die "no txt files created"
echo "extracted $txt_count chapter(s)"

echo ""
echo "--- test: cast ---"
autiobook cast "$WORKDIR" || die "cast command failed"

echo ""
echo "--- test: introduce ---"
autiobook introduce "$WORKDIR" || die "introduce command failed"

# verify introduce output (base voice files have no emotion suffix)
intro_count=$(find "$WORKDIR/introduce" -maxdepth 1 -name "*.wav" | wc -l)
[[ $intro_count -gt 0 ]] || die "no introduce wavs created"
echo "introduced $intro_count character(s)"

echo ""
echo "--- test: audition ---"
autiobook audition "$WORKDIR" || die "audition command failed"

# verify audition output (per-emotion variants have __ suffix)
audition_count=$(find "$WORKDIR/audition" -maxdepth 1 -name "*__*.wav" | wc -l)
[[ $audition_count -gt 0 ]] || die "no audition emotion wavs created"
echo "auditioned $audition_count emotion variant(s)"

echo ""
echo "--- test: script ---"
autiobook script "$WORKDIR" || die "script command failed"

echo ""
echo "--- test: revise ---"
autiobook revise "$WORKDIR" || die "revise command failed"

echo ""
echo "--- test: perform ---"
autiobook perform "$WORKDIR" || die "synthesize command failed"

# verify perform output
wav_count=$(find "$WORKDIR/perform" -name "*.wav" | wc -l)
[[ $wav_count -gt 0 ]] || die "no wav files created"
echo "dramatized $wav_count chapter(s)"

echo ""
echo "--- test: export ---"
autiobook export "$WORKDIR" -o "$OUTDIR" || die "export command failed"

# verify export output
[[ -d "$OUTDIR" ]] || die "output directory not created"
mp3_count=$(find "$OUTDIR" -name "*.mp3" | wc -l)
[[ $mp3_count -gt 0 ]] || die "no mp3 files created"
echo "exported $mp3_count chapter(s)"

# verify mp3 is playable
first_mp3=$(find "$OUTDIR" -name "*.mp3" | sort | head -1)
if command -v ffprobe &>/dev/null; then
	ffprobe -v error "$first_mp3" || die "mp3 file not valid"
	echo "mp3 validated with ffprobe"
fi

echo ""
echo "--- test: idempotency ---"
# run extract again, should not fail
autiobook extract "$TEST_EPUB" -o "$WORKDIR" || die "idempotent extract failed"
# run cast again, should skip existing
autiobook cast "$WORKDIR" || die "idempotent cast failed"
# run introduce again, should skip existing
autiobook introduce "$WORKDIR" || die "idempotent introduce failed"
# run audition again, should skip existing
autiobook audition "$WORKDIR" || die "idempotent audition failed"
# run script again, should skip existing
autiobook script "$WORKDIR" || die "idempotent script failed"
# run perform again, should skip existing
autiobook perform "$WORKDIR" || die "idempotent perform failed"
# run export again, should skip existing
autiobook export "$WORKDIR" -o "$OUTDIR" || die "idempotent export failed"
echo "idempotency check passed"

echo ""
echo "--- test: dramatize (full pipeline) ---"
cleanup
autiobook dramatize "$TEST_EPUB" -o "$WORKDIR" -c 1 || die "dramatize command failed"
[[ -f "$WORKDIR/extract/metadata.json" ]] || die "dramatize: metadata.json not created"
mp3_count=$(find "$OUTDIR" -name "*.mp3" 2>/dev/null | wc -l)
[[ $mp3_count -gt 0 ]] || die "convert: no mp3 files created"
echo "full pipeline completed"

# cleanup
cleanup

echo ""
echo "=== all tests passed ==="
