#!/usr/bin/env bash
#
# extract_blobs.sh - Pull a printer model's calibration tables into src/brother_xl2hb/.
#
# The driver needs the printer's calibration tables (colour LUTs and dither
# tables) to drive the hardware. Those tables ship inside the official
# Brother LPR driver and are not redistributed here — this script downloads the
# `.deb`, verifies its SHA-256, and copies the tables into
# src/brother_xl2hb/lut/<model>/ and src/brother_xl2hb/color_data/<model>/.
# Both directories are gitignored. The unpacked package stays in
# .brother-blobs/<model>/extracted/ for the comparison tools.
#
# Usage:  ./scripts/extract_blobs.sh [model] [--force]
#         model: hl4150cdn (default)
#         (re-runs are idempotent; use --force to redownload)
set -euo pipefail

MODEL=hl4150cdn
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -*) echo "Usage: $0 [model] [--force]" >&2; exit 2 ;;
        *) MODEL="$arg" ;;
    esac
done

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LUT_DIR="$REPO_DIR/src/brother_xl2hb/lut/$MODEL"
COLOR_DATA_DIR="$REPO_DIR/src/brother_xl2hb/color_data/$MODEL"
WORK_DIR="$REPO_DIR/.brother-blobs/$MODEL"

case "$MODEL" in
hl4150cdn)
    DEB_URL="https://download.brother.com/welcome/dlf005940/hl4150cdnlpr-1.1.1-5.i386.deb"
    DEB_FILE="$WORK_DIR/hl4150cdnlpr-1.1.1-5.i386.deb"
    DEB_SHA256="31897d727277311f5f84a15bfd6fcb983b4e658790aeefa4284d1803dd9d0cc4"

    FILTER_BIN_PATH="usr/local/Brother/Printer/hl4150cdn/lpd/brhl4150cdnfilter"
    LUT_PATH_IN_DEB="usr/local/Brother/Printer/hl4150cdn/inf/lut"

    # Single-blob offsets in brhl4150cdnfilter (verified against driver version 1.1.1-5).
    # name=offset:size
    # Colour LUTs are named <profile>[_ig]_<variant>_lut.bin after the tables
    # `lookup_color_transform_table` picks from: profile rgb (Normal), srgb
    # (Vivid) or cmyk (colour matching None); _ig for BRGray=ON; variant density2
    # (toner save), glossy (glossy media) or default.
    BIN_BLOBS=(
      "rgb_default_lut.bin=0x92d58:39304"
      "rgb_density2_lut.bin=0xafb18:39304"
      "rgb_glossy_lut.bin=0xa6138:39304"
      "rgb_ig_default_lut.bin=0xf3057:39304"
      "rgb_ig_density2_lut.bin=0x10fe17:39304"
      "rgb_ig_glossy_lut.bin=0x106437:39304"
      "srgb_default_lut.bin=0xc2ed7:39304"
      "srgb_density2_lut.bin=0xdfc77:39304"
      "srgb_glossy_lut.bin=0xd6297:39304"
      "srgb_ig_default_lut.bin=0x1231d6:39304"
      "srgb_ig_density2_lut.bin=0x13ff76:39304"
      "srgb_ig_glossy_lut.bin=0x136596:39304"
      "cmyk_default_lut.bin=0x32543:39304"
      "cmyk_density2_lut.bin=0x4f343:39304"
      "cmyk_glossy_lut.bin=0x45943:39304"
      "cmyk_ig_default_lut.bin=0x62742:39304"
      "cmyk_ig_density2_lut.bin=0x7f542:39304"
      "cmyk_ig_glossy_lut.bin=0x75b42:39304"
    )

    # interp_tables.bin is 17 sub-tables (17×17×9 = 2601 bytes each) stored at a
    # 0xA40 (2624-byte) stride in the binary — extract and concatenate them.
    INTERP_BASE_OFFSET=0x1bde0
    INTERP_SUB_SIZE=2601
    INTERP_SUB_STRIDE=0xa40
    INTERP_SUB_COUNT=17
    ;;
*)
    echo "ERROR: unknown model $MODEL (known: hl4150cdn)" >&2
    exit 2
    ;;
esac

# Until 1.2.0 the package was downloaded to .brother-blobs/ directly.
OLD_DEB="$REPO_DIR/.brother-blobs/$(basename "$DEB_FILE")"
mkdir -p "$WORK_DIR"
if [[ -f "$OLD_DEB" && ! -f "$DEB_FILE" ]]; then
    mv "$OLD_DEB" "$DEB_FILE"
fi
rm -rf "$REPO_DIR/.brother-blobs/extracted"

mkdir -p "$WORK_DIR" "$LUT_DIR" "$COLOR_DATA_DIR"

# --- 1. Download the LPR deb ------------------------------------------------

if [[ -f "$DEB_FILE" && $FORCE -eq 0 ]]; then
    echo "[1/4] $DEB_FILE already present (use --force to redownload)"
else
    echo "[1/4] Downloading $DEB_URL"
    curl -sSL --fail -o "$DEB_FILE" "$DEB_URL"
fi

# --- 2. Verify SHA-256 ------------------------------------------------------

actual_sha256=""
if command -v sha256sum >/dev/null 2>&1; then
    actual_sha256=$(sha256sum "$DEB_FILE" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
    actual_sha256=$(shasum -a 256 "$DEB_FILE" | awk '{print $1}')
else
    echo "ERROR: neither sha256sum nor shasum found" >&2
    exit 1
fi
if [[ "$actual_sha256" != "$DEB_SHA256" ]]; then
    echo "ERROR: SHA-256 mismatch on $DEB_FILE" >&2
    echo "  expected $DEB_SHA256"     >&2
    echo "  got      $actual_sha256"  >&2
    echo "Refusing to extract — the offsets in this script were computed against the official $(basename "$DEB_FILE")." >&2
    exit 1
fi
echo "[2/4] SHA-256 OK ($DEB_SHA256)"

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
echo "[3/4] Copied $lut_count BRCD dither tables → src/brother_xl2hb/lut/$MODEL/"

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
echo "[4/4] Extracted $total_blobs calibration blobs → src/brother_xl2hb/color_data/$MODEL/"

echo ""
echo "Done. The driver can now reproduce byte-identical Brother output."
echo "--force re-downloads and re-extracts the pinned driver package."
