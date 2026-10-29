#!/bin/bash
#
# Install the open-source CUPS driver for one Brother printer model
# (default: the HL-4150CDN). Requires: sudo, Python 3.13+, Ghostscript
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# CUPS paths (Linux). macOS is not supported: its CUPS only runs filters from
# /usr/libexec/cups/filter and ships PPDs under /usr/share/cups/model, both on
# the SIP-protected system volume.
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Error: macOS is not supported by this installer (the CUPS filter and PPD"
    echo "  directories are SIP-protected). Install the driver on a Linux print server."
    exit 1
fi
CUPS_PPD_DIR="/usr/share/cups/model"
CUPS_FILTER_DIR="/usr/lib/cups/filter"

MODEL="hl4150cdn"
ADD_PRINTER=false
PRINTER_URI=""

usage() {
    echo "Usage: $0 [--model MODEL] [--add-printer [URI]]"
    echo ""
    echo "Options:"
    echo "  --model MODEL        Printer model to install (hl4150cdn, the default)."
    echo "  --add-printer [URI]  Create a CUPS printer queue after installing."
    echo "                       If URI is omitted, attempts auto-detection via"
    echo "                       lpinfo or falls back to socket://BRW.local:9100"
    echo ""
    echo "This script must be run with sudo."
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="${2:?--model needs a value}"
            shift 2
            ;;
        --add-printer)
            ADD_PRINTER=true
            if [[ ${2:-} && ! ${2:-} == --* ]]; then
                PRINTER_URI="$2"
                shift
            fi
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Per-model names: the model's PPD and CUPS filter are cups/br<model>.ppd and
# cups/br<model>-filter; it installs to /usr/local/lib/br<model>/.
case "$MODEL" in
    hl4150cdn)
        MODEL_NAME="HL-4150CDN"
        DETECT_PATTERN="hl-4150"
        ;;
    *)
        echo "Error: unknown model $MODEL (known: hl4150cdn)"
        exit 1
        ;;
esac
DRIVER="br$MODEL"
FILTER="$DRIVER-filter"
PPD="$DRIVER.ppd"
QUEUE="Brother_$MODEL_NAME"

# Installation paths
LIB_DIR="/usr/local/lib/$DRIVER"
VENV_DIR="$LIB_DIR/.venv"
PKG_SRC="$REPO_ROOT/src/brother_xl2hb"
PKG_DIR="$LIB_DIR/brother_xl2hb"

# Check root
if [[ $EUID -ne 0 ]]; then
    echo "Error: This script must be run with sudo."
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Please install Python 3.13+."
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [[ $PY_MAJOR -lt 3 ]] || [[ $PY_MAJOR -eq 3 && $PY_MINOR -lt 13 ]]; then
    echo "Error: Python 3.13+ required, found $PY_VERSION"
    exit 1
fi
echo "  Python $PY_VERSION: OK"

if ! command -v gs &>/dev/null; then
    echo "Warning: Ghostscript (gs) not found. Install it before printing."
    echo "  sudo apt install ghostscript"
fi

# Create library directory
echo "Installing library files to $LIB_DIR..."
mkdir -p "$LIB_DIR"

# Replace the previous installation; up to 1.2.0 the modules and data sat
# flat in $LIB_DIR. The inverse LUTs are recomputed below anyway.
rm -rf "$PKG_DIR" "$LIB_DIR/color_data" "$LIB_DIR/lut" "$LIB_DIR/__pycache__"
rm -f "$LIB_DIR/"*.py "$LIB_DIR/"_*_fast*.so

# Copy the package with the tables extract_blobs.sh put into it
cp -R "$PKG_SRC" "$PKG_DIR"
find "$PKG_DIR" \( -name __pycache__ -o -name '*.so' -o -name '*.c' -o -name 'inverse_lut_*.npy' \) -prune -exec rm -rf {} +
mkdir -p "$PKG_DIR/color_data/$MODEL"

if ! compgen -G "$PKG_DIR/lut/$MODEL/*.bin" >/dev/null; then
    echo "Warning: BRCD LUT files not found in $PKG_SRC/lut/$MODEL"
    echo "  Run scripts/extract_blobs.sh $MODEL to fetch them from Brother's LPR package."
    echo "  The driver will fall back to built-in Bayer dithering otherwise."
fi

# Create virtual environment with dependencies
echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet "numpy>=2.5.3"  # keep in sync with pyproject.toml

# Precompute the RGB→KCMY inverse LUTs (64 MiB each: Normal and Vivid colour
# matching, each with and without improve gray). Replaces per-pixel tetrahedral
# interpolation. The file names carry a hash of the colour grids they were
# computed from; LUTs from older grids (e.g. before extract_blobs.sh) are
# removed here and ignored by the filter.
echo "Precomputing inverse color LUTs (takes about four minutes on a Pi 3)..."
if ! (cd "$LIB_DIR" && PYTHONPATH="$LIB_DIR" "$VENV_DIR/bin/python3" -m brother_xl2hb.cli --model "$MODEL" --precompute-lut); then
    echo "Warning: inverse LUT precompute failed; falling back to per-pixel interpolation."
fi

# Optional Cython build of the RLE encoders, the colour-LUT gather, the band
# renderer (colour + dither + RLE per band, renders a page on several cores)
# and its Fine mode counterpart with the Fine codec.
# Needs a C compiler and the Python headers (Debian: python3-dev); pure-Python
# fallback otherwise.
echo "Building Cython acceleration..."
BUILD_DIR="$(mktemp -d)"
NATIVE_BUILD="$BUILD_DIR/src/brother_xl2hb/_native"
mkdir -p "$NATIVE_BUILD"
cp "$REPO_ROOT/setup_cython.py" "$BUILD_DIR/"
cp "$PKG_SRC/__init__.py" "$BUILD_DIR/src/brother_xl2hb/"
cp "$PKG_SRC/_native/"*.py "$PKG_SRC/_native/"*.pyx "$PKG_SRC/_native/"*.pxd "$NATIVE_BUILD/"
if "$VENV_DIR/bin/pip" install --quiet cython setuptools \
    && (cd "$BUILD_DIR" && "$VENV_DIR/bin/python3" setup_cython.py build_ext --inplace >/dev/null) \
    && compgen -G "$NATIVE_BUILD/_*_fast*.so" >/dev/null; then
    cp "$NATIVE_BUILD/"_*_fast*.so "$PKG_DIR/_native/"
    echo "  Cython extensions installed."
else
    echo "Warning: Cython build failed (missing compiler or python3-dev?); using pure-Python fallbacks."
fi
rm -rf "$BUILD_DIR"

# Install CUPS filter
echo "Installing CUPS filter to $CUPS_FILTER_DIR..."
mkdir -p "$CUPS_FILTER_DIR"

# Create a wrapper script that activates the venv
cat > "$CUPS_FILTER_DIR/$FILTER" << WRAPPER_EOF
#!/bin/bash
# CUPS filter wrapper for Brother $MODEL_NAME (Open Source)
# Activates the venv and runs the Python filter.
LIB_DIR="$LIB_DIR"
VENV_DIR="\$LIB_DIR/.venv"
export PATH="\$VENV_DIR/bin:\$PATH"
export PYTHONPATH="\$LIB_DIR"
exec "\$VENV_DIR/bin/python3" "\$LIB_DIR/$FILTER.py" "\$@"
WRAPPER_EOF
chmod 755 "$CUPS_FILTER_DIR/$FILTER"

# Copy the actual filter script to the lib dir
cp "$SCRIPT_DIR/$FILTER" "$LIB_DIR/$FILTER.py"
chmod 644 "$LIB_DIR/$FILTER.py"

# Precompile bytecode: the CUPS filter user cannot write __pycache__ here,
# so without this every job recompiles all modules.
"$VENV_DIR/bin/python3" -m compileall -q "$PKG_DIR" "$LIB_DIR/$FILTER.py"

# Install PPD
echo "Installing PPD to $CUPS_PPD_DIR..."
mkdir -p "$CUPS_PPD_DIR"
cp "$SCRIPT_DIR/$PPD" "$CUPS_PPD_DIR/$PPD"
chmod 644 "$CUPS_PPD_DIR/$PPD"

echo ""
echo "Installation complete!"
echo ""

# Optionally add printer
if $ADD_PRINTER; then
    echo "Setting up printer queue..."

    # Auto-detect printer URI if not specified
    if [[ -z "$PRINTER_URI" ]]; then
        echo "  Searching for printer..."
        DETECTED=$(lpinfo -v 2>/dev/null | grep -i "$DETECT_PATTERN" | head -1 | awk '{print $2}' || true)
        if [[ -n "$DETECTED" ]]; then
            PRINTER_URI="$DETECTED"
            echo "  Found: $PRINTER_URI"
        else
            # Try common network name
            PRINTER_URI="socket://BRW.local:9100"
            echo "  Not auto-detected, using: $PRINTER_URI"
            echo "  You may need to update the URI with:"
            echo "    lpadmin -p $QUEUE -v <actual-uri>"
        fi
    fi

    lpadmin -p "$QUEUE" \
        -E \
        -v "$PRINTER_URI" \
        -P "$CUPS_PPD_DIR/$PPD" \
        -D "Brother $MODEL_NAME (Open Source)" \
        -L "Network Printer"

    echo "  Printer queue '$QUEUE' created."
    echo "  Test with: lp -d $QUEUE testpage.pdf"
else
    echo "To add a printer queue, run:"
    echo "  sudo lpadmin -p $QUEUE -E \\"
    echo "    -v socket://YOUR_PRINTER_IP:9100 \\"
    echo "    -P $CUPS_PPD_DIR/$PPD \\"
    echo "    -D 'Brother $MODEL_NAME (Open Source)'"
    echo ""
    echo "Or re-run this script with --add-printer:"
    echo "  sudo $0 --model $MODEL --add-printer"
fi
