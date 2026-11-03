#!/usr/bin/env bash
#
# build_rust.sh - Build the Rust kernels into src/brother_xl2hb/_native/_rust.abi3.so.
#
# The crate in rust/ builds one abi3 extension for every Python >= 3.13.
# `brother_xl2hb._native.load` prefers its kernels over the Cython ones
# (BRXL2HB_BACKEND picks a backend explicitly).
#
# Usage:  ./scripts/build_rust.sh
#         needs cargo; PyO3 reads its Python from PYO3_PYTHON, by default
#         .venv/bin/python if present, else python3.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="$REPO_ROOT/rust"
DEST="$REPO_ROOT/src/brother_xl2hb/_native/_rust.abi3.so"

if [[ -z "${PYO3_PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYO3_PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        PYO3_PYTHON="$(command -v python3)"
    fi
fi
export PYO3_PYTHON

# An explicit target dir, so a CARGO_TARGET_DIR elsewhere cannot leave an
# old library at the path copied below.
TARGET="$CRATE/target"
cargo build --release --manifest-path "$CRATE/Cargo.toml" --target-dir "$TARGET"

case "$(uname -s)" in
    Darwin) LIB="$TARGET/release/lib_rust.dylib" ;;
    *) LIB="$TARGET/release/lib_rust.so" ;;
esac
cp "$LIB" "$DEST"
echo "Installed $DEST"
