#!/bin/bash
#
# Install the open-source Brother HL-4150CDN CUPS driver.
# Requires: sudo, Python 3.13+, Ghostscript
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Installation paths
LIB_DIR="/usr/local/lib/brhl4150cdn"
VENV_DIR="$LIB_DIR/.venv"

# Detect OS-specific CUPS paths
if [[ "$(uname)" == "Darwin" ]]; then
    CUPS_FILTER_DIR="/usr/libexec/cups/filter"
    CUPS_PPD_DIR="/usr/share/cups/model"
else
    CUPS_FILTER_DIR="/usr/lib/cups/filter"
    CUPS_PPD_DIR="/usr/share/cups/model"
fi

ADD_PRINTER=false
PRINTER_URI=""

usage() {
    echo "Usage: $0 [--add-printer [URI]]"
    echo ""
    echo "Options:"
    echo "  --add-printer [URI]  Create a CUPS printer queue after installing."
    echo "                       If URI is omitted, attempts auto-detection via"
    echo "                       lpinfo or falls back to socket://BRW*.local:9100"
    echo ""
    echo "This script must be run with sudo."
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
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
    echo "  macOS: brew install ghostscript"
    echo "  Linux: sudo apt install ghostscript"
fi

# Create library directory
echo "Installing library files to $LIB_DIR..."
mkdir -p "$LIB_DIR"

# Copy Python source files (all modules in src/)
cp "$REPO_ROOT/src/"*.py "$LIB_DIR/"

# Copy color data
mkdir -p "$LIB_DIR/color_data"
cp "$REPO_ROOT/src/color_data/"*.bin "$LIB_DIR/color_data/"

# Copy BRCD LUT files (extracted by scripts/extract_blobs.sh into src/lut/)
LUT_SRC="$REPO_ROOT/src/lut"
if [[ -d "$LUT_SRC" ]] && compgen -G "$LUT_SRC/*.bin" >/dev/null; then
    echo "Installing BRCD dither tables..."
    mkdir -p "$LIB_DIR/lut"
    cp "$LUT_SRC/"*.bin "$LIB_DIR/lut/"
else
    echo "Warning: BRCD LUT files not found at $LUT_SRC"
    echo "  Run scripts/extract_blobs.sh to fetch them from Brother's LPR package."
    echo "  The driver will fall back to built-in Bayer dithering otherwise."
fi

# Create virtual environment with dependencies
echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --quiet numpy

# Install CUPS filter
echo "Installing CUPS filter to $CUPS_FILTER_DIR..."
mkdir -p "$CUPS_FILTER_DIR"

# Create a wrapper script that activates the venv
cat > "$CUPS_FILTER_DIR/brhl4150cdn-filter" << 'WRAPPER_EOF'
#!/bin/bash
# CUPS filter wrapper for Brother HL-4150CDN (Open Source)
# Activates the venv and runs the Python filter.
LIB_DIR="/usr/local/lib/brhl4150cdn"
VENV_DIR="$LIB_DIR/.venv"
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONPATH="$LIB_DIR"
exec "$VENV_DIR/bin/python3" "$LIB_DIR/brhl4150cdn-filter.py" "$@"
WRAPPER_EOF
chmod 755 "$CUPS_FILTER_DIR/brhl4150cdn-filter"

# Copy the actual filter script to the lib dir
cp "$SCRIPT_DIR/brhl4150cdn-filter" "$LIB_DIR/brhl4150cdn-filter.py"
chmod 644 "$LIB_DIR/brhl4150cdn-filter.py"

# Install PPD
echo "Installing PPD to $CUPS_PPD_DIR..."
mkdir -p "$CUPS_PPD_DIR"
cp "$SCRIPT_DIR/brhl4150cdn.ppd" "$CUPS_PPD_DIR/brhl4150cdn.ppd"
chmod 644 "$CUPS_PPD_DIR/brhl4150cdn.ppd"

echo ""
echo "Installation complete!"
echo ""

# Optionally add printer
if $ADD_PRINTER; then
    echo "Setting up printer queue..."

    # Auto-detect printer URI if not specified
    if [[ -z "$PRINTER_URI" ]]; then
        echo "  Searching for printer..."
        DETECTED=$(lpinfo -v 2>/dev/null | grep -i "hl-4150" | head -1 | awk '{print $2}' || true)
        if [[ -n "$DETECTED" ]]; then
            PRINTER_URI="$DETECTED"
            echo "  Found: $PRINTER_URI"
        else
            # Try common network name
            PRINTER_URI="socket://BRW.local:9100"
            echo "  Not auto-detected, using: $PRINTER_URI"
            echo "  You may need to update the URI with:"
            echo "    lpadmin -p Brother_HL-4150CDN -v <actual-uri>"
        fi
    fi

    lpadmin -p Brother_HL-4150CDN \
        -E \
        -v "$PRINTER_URI" \
        -P "$CUPS_PPD_DIR/brhl4150cdn.ppd" \
        -D "Brother HL-4150CDN (Open Source)" \
        -L "Network Printer"

    echo "  Printer queue 'Brother_HL-4150CDN' created."
    echo "  Test with: lp -d Brother_HL-4150CDN testpage.pdf"
else
    echo "To add a printer queue, run:"
    echo "  sudo lpadmin -p Brother_HL-4150CDN -E \\"
    echo "    -v socket://YOUR_PRINTER_IP:9100 \\"
    echo "    -P $CUPS_PPD_DIR/brhl4150cdn.ppd \\"
    echo "    -D 'Brother HL-4150CDN (Open Source)'"
    echo ""
    echo "Or re-run this script with --add-printer:"
    echo "  sudo $0 --add-printer"
fi
