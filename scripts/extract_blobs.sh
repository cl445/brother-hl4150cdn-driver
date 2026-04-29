#!/usr/bin/env bash
#
# extract_blobs.sh - Pull Brother HL-4150CDN calibration tables into src/.
#
# The driver needs the printer's calibration tables (colour LUTs, gamma curves,
# dither tables) to drive the hardware. Those tables ship inside the official
# Brother LPR driver and are not redistributed here — this script downloads the
# `.deb`, verifies its MD5, and copies the tables into src/lut/ and
# src/color_data/. Both directories are gitignored.
#
# Usage:  ./scripts/extract_blobs.sh
#         (re-runs are idempotent; use --force to redownload)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LUT_DIR="$REPO_DIR/src/lut"
COLOR_DATA_DIR="$REPO_DIR/src/color_data"
WORK_DIR="$REPO_DIR/.brother-blobs"

DEB_URL="https://download.brother.com/welcome/dlf005940/hl4150cdnlpr-1.1.1-5.i386.deb"
DEB_FILE="$WORK_DIR/hl4150cdnlpr-1.1.1-5.i386.deb"
DEB_MD5="dd803993d4a29a432d63141201821b82"

FILTER_BIN_PATH="usr/local/Brother/Printer/hl4150cdn/lpd/brhl4150cdnfilter"
LUT_PATH_IN_DEB="usr/local/Brother/Printer/hl4150cdn/inf/lut"

# Single-blob offsets in brhl4150cdnfilter (verified against driver version 1.1.1-5).
# name=offset:size
BIN_BLOBS=(
  "rgb_default_lut.bin=0x92d58:39304"
  "srgb_default_lut.bin=0xc2ed7:39304"
  "gamma_curve_0.bin=0x283c0:256"
  "gamma_curve_1.bin=0x284c0:256"
)

# interp_tables.bin is 17 sub-tables (17×17×9 = 2601 bytes each) stored at a
# 0xA40 (2624-byte) stride in the binary — extract and concatenate them.
INTERP_BASE_OFFSET=0x1bde0
INTERP_SUB_SIZE=2601
INTERP_SUB_STRIDE=0xa40
INTERP_SUB_COUNT=17

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

mkdir -p "$WORK_DIR" "$LUT_DIR" "$COLOR_DATA_DIR"

# --- 1. Download the LPR deb ------------------------------------------------

if [[ -f "$DEB_FILE" && $FORCE -eq 0 ]]; then
    echo "[1/4] $DEB_FILE already present (use --force to redownload)"
else
    echo "[1/4] Downloading $DEB_URL"
    curl -sSL --fail -o "$DEB_FILE" "$DEB_URL"
fi

# --- 2. Verify MD5 ----------------------------------------------------------

actual_md5=""
if command -v md5sum >/dev/null 2>&1; then
    actual_md5=$(md5sum "$DEB_FILE" | awk '{print $1}')
elif command -v md5 >/dev/null 2>&1; then
    actual_md5=$(md5 -q "$DEB_FILE")
else
    echo "ERROR: neither md5sum nor md5 found" >&2
    exit 1
fi
if [[ "$actual_md5" != "$DEB_MD5" ]]; then
    echo "ERROR: MD5 mismatch on $DEB_FILE" >&2
    echo "  expected $DEB_MD5"     >&2
    echo "  got      $actual_md5"  >&2
    echo "Refusing to extract — the offsets in this script were computed against the official 1.1.1-5 build." >&2
    exit 1
fi
echo "[2/4] MD5 OK ($DEB_MD5)"

# --- 3. Unpack the deb and copy the BRCD dither tables ---------------------

EXTRACT_DIR="$WORK_DIR/extracted"
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
( cd "$EXTRACT_DIR" && ar x "$DEB_FILE" data.tar.gz && tar xzf data.tar.gz && rm data.tar.gz )

src_lut="$EXTRACT_DIR/$LUT_PATH_IN_DEB"
if [[ ! -d "$src_lut" ]]; then
    echo "ERROR: expected $src_lut inside the deb" >&2
    exit 1
fi
cp "$src_lut"/*.bin "$LUT_DIR/"
lut_count=$(find "$LUT_DIR" -maxdepth 1 -name '*.bin' | wc -l | tr -d ' ')
echo "[3/4] Copied $lut_count BRCD dither tables → src/lut/"

# --- 4. Extract calibration blobs from the ELF binary ----------------------

filter_bin="$EXTRACT_DIR/$FILTER_BIN_PATH"
if [[ ! -f "$filter_bin" ]]; then
    echo "ERROR: $filter_bin not found" >&2
    exit 1
fi

for entry in "${BIN_BLOBS[@]}"; do
    name="${entry%%=*}"
    rest="${entry#*=}"
    offset_hex="${rest%%:*}"
    size="${rest##*:}"
    offset_dec=$((offset_hex))
    out="$COLOR_DATA_DIR/$name"
    dd if="$filter_bin" of="$out" bs=1 skip="$offset_dec" count="$size" status=none
    actual_size=$(wc -c < "$out" | tr -d ' ')
    if [[ "$actual_size" != "$size" ]]; then
        echo "ERROR: $name extracted $actual_size bytes (expected $size)" >&2
        exit 1
    fi
done

# interp_tables.bin: concatenate 17 sub-tables read from a strided layout.
interp_out="$COLOR_DATA_DIR/interp_tables.bin"
: > "$interp_out"
for ((i = 0; i < INTERP_SUB_COUNT; i++)); do
    sub_offset=$((INTERP_BASE_OFFSET + i * INTERP_SUB_STRIDE))
    dd if="$filter_bin" bs=1 skip="$sub_offset" count="$INTERP_SUB_SIZE" status=none >> "$interp_out"
done
expected_interp_size=$((INTERP_SUB_COUNT * INTERP_SUB_SIZE))
actual_interp_size=$(wc -c < "$interp_out" | tr -d ' ')
if [[ "$actual_interp_size" != "$expected_interp_size" ]]; then
    echo "ERROR: interp_tables.bin extracted $actual_interp_size bytes (expected $expected_interp_size)" >&2
    exit 1
fi

total_blobs=$((${#BIN_BLOBS[@]} + 1))
echo "[4/4] Extracted $total_blobs calibration blobs → src/color_data/"

echo ""
echo "Done. The driver can now reproduce byte-identical Brother output."
echo "Re-run with --force after upgrading Brother's driver version."
