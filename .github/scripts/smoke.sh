#!/usr/bin/env bash
# Shared end-to-end smoke test for the decoder, used by every CI job on both
# Windows (git-bash) and Linux. The only platform difference is the binary
# name, which the caller passes as $1.
#
# Usage: bash .github/scripts/smoke.sh target/release/ld-decode[.exe]
#
# This is NOT the parity gate (that is b3sum 4/4 against the reference decodes,
# which needs the capture files and hours of runtime). It proves the binary
# starts, parses its arguments, runs the reader/no-signal/writer paths to
# completion through every output format, and exits cleanly.
set -euo pipefail

BIN="${1:?usage: smoke.sh <path to ld-decode binary>}"
WORK="${2:-smoke}"

if [ ! -f "$BIN" ]; then
  echo "No decoder binary at $BIN" >&2
  exit 1
fi

rm -rf "$WORK"
mkdir -p "$WORK"

"$BIN" --help > "$WORK/help.txt"
grep -q -- "--start" "$WORK/help.txt"
grep -q -- "--threads" "$WORK/help.txt"

# One second of zero samples. Nothing decodes, but the reader, the no-signal
# recovery path and every writer (tbc/pcm/efm/json/db/log) must still run to
# completion and exit cleanly.
head -c 40000000 /dev/zero > "$WORK/zeros.s16"
"$BIN" -j 4 -l 4 "$WORK/zeros.s16" "$WORK/out"

for ext in tbc pcm efm tbc.json tbc.db log; do
  if [ ! -f "$WORK/out.$ext" ]; then
    echo "Missing output: $WORK/out.$ext" >&2
    ls -la "$WORK"
    exit 1
  fi
done
grep -q '"fields":\[\]' "$WORK/out.tbc.json"
grep -q "Decode finished" "$WORK/out.log"

# Container sniffing for `.ldf`. The extension normally carries Ogg-wrapped
# FLAC (what the Domesday Duplicator writes), but the Python reference decodes
# these inputs with PyAV, which detects the container itself -- so a bare FLAC
# stream under `.ldf` has to decode too, instead of failing with "No Ogg
# capture pattern found". (ffmpeg is the same decoder the `.flac` path spawns.)
ffmpeg -v error -y -f s16le -ar 40000 -ac 1 -i "$WORK/zeros.s16" \
  -c:a flac -f flac "$WORK/zeros.ldf"
cp "$WORK/zeros.ldf" "$WORK/zeros.flac"
"$BIN" -j 4 -l 4 "$WORK/zeros.ldf" "$WORK/ldf"
"$BIN" -j 4 -l 4 "$WORK/zeros.flac" "$WORK/flac"
cmp "$WORK/flac.tbc" "$WORK/ldf.tbc"

# Ogg-wrapped FLAC must still take the Ogg path.
ffmpeg -v error -y -f s16le -ar 40000 -ac 1 -i "$WORK/zeros.s16" \
  -c:a flac -compression_level 6 -f ogg "$WORK/ogg.ldf"
"$BIN" -j 4 -l 4 "$WORK/ogg.ldf" "$WORK/ogg"
for ext in tbc pcm efm tbc.json tbc.db log; do
  if [ ! -f "$WORK/ogg.$ext" ]; then
    echo "Missing output: $WORK/ogg.$ext" >&2
    exit 1
  fi
done

# A file that is not FLAC at all fails on the sniffing message.
head -c 100000 /dev/urandom > "$WORK/junk.ldf"
if "$BIN" -j 4 -l 4 "$WORK/junk.ldf" "$WORK/junk" > /dev/null 2> "$WORK/junk.err"; then
  echo "junk.ldf unexpectedly decoded" >&2
  exit 1
fi
grep -q "no FLAC capture found" "$WORK/junk.err"

# `-` streams the picture to stdout: the bytes must match the file writer's
# exactly, no sidecar files may appear, and the console log must move to stderr.
"$BIN" -j 4 -l 4 "$WORK/zeros.s16" - > "$WORK/pipe.tbc" 2> "$WORK/pipe.err"
cmp "$WORK/out.tbc" "$WORK/pipe.tbc"
if compgen -G "$WORK/-.*" > /dev/null; then
  echo "Pipe mode created sidecar files:" >&2
  compgen -G "$WORK/-.*"
  exit 1
fi
grep -q "streaming the .tbc picture to stdout" "$WORK/pipe.err"

echo "Smoke test outputs:"
ls -la "$WORK"
echo "Smoke test OK: $BIN"
