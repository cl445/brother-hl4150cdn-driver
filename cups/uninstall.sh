#!/bin/bash
#
# Uninstall the open-source CUPS driver for one Brother printer model
# (default: the HL-4150CDN).
#
set -euo pipefail

# CUPS paths (Linux; install.sh does not support macOS)
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Error: macOS is not supported (install.sh never installs there)."
    exit 1
fi
CUPS_PPD_DIR="/usr/share/cups/model"
CUPS_FILTER_DIR="/usr/lib/cups/filter"

MODEL="hl4150cdn"
REMOVE_PRINTER=false

usage() {
    echo "Usage: $0 [--model MODEL] [--remove-printer]"
    echo ""
    echo "Options:"
    echo "  --model MODEL     Printer model to uninstall (hl4150cdn, the default)."
    echo "  --remove-printer  Also remove the CUPS printer queue"
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
        --remove-printer)
            REMOVE_PRINTER=true
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

case "$MODEL" in
    hl4150cdn) MODEL_NAME="HL-4150CDN" ;;
    *)
        echo "Error: unknown model $MODEL (known: hl4150cdn)"
        exit 1
        ;;
esac
DRIVER="br$MODEL"
FILTER="$DRIVER-filter"
PPD="$DRIVER.ppd"
QUEUE="Brother_$MODEL_NAME"
LIB_DIR="/usr/local/lib/$DRIVER"

echo "Uninstalling Brother $MODEL_NAME open-source driver..."

# Remove printer queue if requested
if $REMOVE_PRINTER; then
    if lpstat -p "$QUEUE" &>/dev/null; then
        echo "Removing printer queue '$QUEUE'..."
        lpadmin -x "$QUEUE"
    else
        echo "Printer queue '$QUEUE' not found (skipping)."
    fi
fi

# Remove CUPS filter
if [[ -f "$CUPS_FILTER_DIR/$FILTER" ]]; then
    echo "Removing CUPS filter..."
    rm -f "$CUPS_FILTER_DIR/$FILTER"
fi

# Remove PPD
if [[ -f "$CUPS_PPD_DIR/$PPD" ]]; then
    echo "Removing PPD..."
    rm -f "$CUPS_PPD_DIR/$PPD"
fi

# Remove library directory
if [[ -d "$LIB_DIR" ]]; then
    echo "Removing library files from $LIB_DIR..."
    rm -rf "$LIB_DIR"
fi

echo ""
echo "Uninstallation complete."

if ! $REMOVE_PRINTER; then
    if lpstat -p "$QUEUE" &>/dev/null; then
        echo ""
        echo "Note: Printer queue '$QUEUE' still exists."
        echo "To remove it, run: sudo $0 --model $MODEL --remove-printer"
        echo "Or manually: sudo lpadmin -x $QUEUE"
    fi
fi
