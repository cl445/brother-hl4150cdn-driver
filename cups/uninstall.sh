#!/bin/bash
#
# Uninstall the open-source Brother HL-4150CDN CUPS driver.
#
set -euo pipefail

# Installation paths
LIB_DIR="/usr/local/lib/brhl4150cdn"

# Detect OS-specific CUPS paths
if [[ "$(uname)" == "Darwin" ]]; then
    CUPS_FILTER_DIR="/usr/libexec/cups/filter"
    CUPS_PPD_DIR="/usr/share/cups/model"
else
    CUPS_FILTER_DIR="/usr/lib/cups/filter"
    CUPS_PPD_DIR="/usr/share/cups/model"
fi

REMOVE_PRINTER=false

usage() {
    echo "Usage: $0 [--remove-printer]"
    echo ""
    echo "Options:"
    echo "  --remove-printer  Also remove the CUPS printer queue"
    echo ""
    echo "This script must be run with sudo."
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
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

echo "Uninstalling Brother HL-4150CDN open-source driver..."

# Remove printer queue if requested
if $REMOVE_PRINTER; then
    if lpstat -p Brother_HL-4150CDN &>/dev/null; then
        echo "Removing printer queue 'Brother_HL-4150CDN'..."
        lpadmin -x Brother_HL-4150CDN
    else
        echo "Printer queue 'Brother_HL-4150CDN' not found (skipping)."
    fi
fi

# Remove CUPS filter
if [[ -f "$CUPS_FILTER_DIR/brhl4150cdn-filter" ]]; then
    echo "Removing CUPS filter..."
    rm -f "$CUPS_FILTER_DIR/brhl4150cdn-filter"
fi

# Remove PPD
if [[ -f "$CUPS_PPD_DIR/brhl4150cdn.ppd" ]]; then
    echo "Removing PPD..."
    rm -f "$CUPS_PPD_DIR/brhl4150cdn.ppd"
fi

# Remove library directory
if [[ -d "$LIB_DIR" ]]; then
    echo "Removing library files from $LIB_DIR..."
    rm -rf "$LIB_DIR"
fi

echo ""
echo "Uninstallation complete."

if ! $REMOVE_PRINTER; then
    if lpstat -p Brother_HL-4150CDN &>/dev/null 2>&1; then
        echo ""
        echo "Note: Printer queue 'Brother_HL-4150CDN' still exists."
        echo "To remove it, run: sudo $0 --remove-printer"
        echo "Or manually: sudo lpadmin -x Brother_HL-4150CDN"
    fi
fi
