# Brother HL-4150CDN Driver

[![CI](https://github.com/cl445/brother-hl4150cdn-driver/actions/workflows/ci.yml/badge.svg)](https://github.com/cl445/brother-hl4150cdn-driver/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

Open-source CUPS driver for the Brother HL-4150CDN colour laser printer.
Brother's own Linux driver runs only on x86 PCs. This one is written in
Python with optional Cython acceleration and runs anywhere CUPS does,
including ARM boards such as the Raspberry Pi.

## Features

- Sends the printer the same data as Brother's driver, byte for byte
- Normal (600 dpi) and Fine (600x2400 dpi) print quality
- Colour and grayscale, simplex and duplex on the long or short edge
- 21 paper sizes, all trays and media types
- Brother's colour settings: colour matching, improve gray, enhance
  black, toner save, brightness, contrast, saturation and per-channel
  colour adjustment
- Reverse print, skip blank pages and copies

On a Raspberry Pi 3B+ a page takes about 1 s in Normal and about 6.5 s in
Fine quality. The filter needs about 100 to 250 MB of memory, depending
on the options.

## Requirements

- Linux with CUPS and Ghostscript (`install.sh` does not support macOS)
- Python 3.13+ and [`uv`](https://docs.astral.sh/uv/)
- Recommended: a C compiler and the Python headers (Debian:
  `sudo apt install build-essential python3-dev`). `install.sh` then
  builds the native modules; without them the driver is several times
  slower, and Fine mode about 30 times.
- About 260 MB of disk space for precomputed colour tables. `install.sh`
  computes them, which takes about four minutes on a Pi 3.
- Brother's HL-4150CDN LPR driver package: `scripts/extract_blobs.sh`
  downloads it and extracts the printer's calibration tables, which this
  repository does not include.

## Install

```bash
git clone https://github.com/cl445/brother-hl4150cdn-driver.git
cd brother-hl4150cdn-driver

uv sync --all-extras
./scripts/extract_blobs.sh
sudo cups/install.sh                                            # filter + PPD
sudo cups/install.sh --add-printer socket://<printer-ip>:9100   # network, optional
```

Pick the device URI that matches how the printer is attached:

| Connection | URI | Discover |
|---|---|---|
| Ethernet / Wi-Fi | `socket://<printer-ip>:9100` | manufacturer's web UI / `nmap -p9100` |
| USB | `usb://Brother/HL-4150CDN?serial=<serial>` | `lpinfo --include-schemes usb -l -v` |

Without `--add-printer`, register the queue manually:

```bash
sudo lpadmin -p Brother_HL-4150CDN -E \
  -v <device-uri> \
  -P /usr/share/cups/model/brhl4150cdn.ppd
```

`sudo cups/uninstall.sh --remove-printer` cleans up.

### Upgrading

Pull, then re-run `./scripts/extract_blobs.sh` and `sudo cups/install.sh`.
Existing queues keep their old copy of the PPD; to pick up new options
(e.g. the standard `Duplex` option since 1.1.0, Fine since 1.2.0),
re-assign it. This resets the queue's default options:

```bash
sudo lpadmin -p Brother_HL-4150CDN -P /usr/share/cups/model/brhl4150cdn.ppd
```

## Settings

All options appear in the standard print dialog. Pass them as
`-o key=value` to `lp` / `lpr` for scripting.

| Option | Values | Notes |
|---|---|---|
| `PageSize` | A4, Letter, Legal, Executive, A5, PRA5Rotated (A5 long edge), A6, ISOB5, ISOB6, JISB5, JISB6, Postcard, EnvDL, EnvPRC5Rotated (DL long edge), EnvC5, Env10 (Com-10), EnvMonarch, Br3x5, FanFoldGermanLegal (Folio), EnvYou4 (Envelope #4), EnvChou3 (Envelope MAX) | |
| `BRMediaType` | Plain, Thin, Thick, Thicker, Bond, Envelope, EnvThin, EnvThick, Recycled, Postcard, Label, Glossy | Glossy also changes the colour calibration |
| `BRResolution` | 600dpi (Normal), 600x2400dpi (Fine) | Fine: finer detail, slower |
| `BRMonoColor` | Auto, FullColor, Mono | |
| `Duplex` | None, DuplexNoTumble, DuplexTumble | Tumble = short edge; IPP `sides` works too |
| `BRColorMatching` | Normal, Vivid, None | None turns colour matching off |
| `BRGray` | OFF, ON | Improve gray |
| `BREnhanceBlkPrt` | OFF, ON | Enhance black: prints pure black as rich black |
| `BRImproveOutput` | OFF, BRLessPaperCurl, BRFixIntensity | |
| `BRInputSlot` | AutoSelect, Tray1, Tray2, MPTray | |
| `BRTonerSaveMode` | OFF, ON | Toner save |
| `BRSkipBlank` | OFF, ON | Skips blank pages, in duplex too |
| `BRReverse` | OFF, ON | Prints each page rotated by 180°, page order unchanged |
| `BRBrightness` | -20 … 20 | |
| `BRContrast` | -20 … 20 | |
| `BRSaturation` | -20 … 20 | |
| `BRRed`, `BRGreen`, `BRBlue` | -20 … 20 | Per-channel colour adjustment |

Copies come from the standard `-n` / `copies` option.

## CLI without CUPS

```bash
# PPM in, XL2HB raw stream out
cat page.ppm | uv run brother-xl2hb \
    --paper Letter --duplex long --toner-save \
  > page.xl2hb

# Send straight to the printer (network)
cat page.ppm | uv run brother-xl2hb | nc <printer-ip> 9100

# …or via USB
cat page.ppm | uv run brother-xl2hb > /dev/usb/lp0
```

## Development

```bash
uv sync --all-extras
uv run nox --list

# common sessions
uv run nox -s lint            # ruff
uv run nox -s format_check    # ruff format --check
uv run nox -s typecheck       # pyrefly
uv run nox -s deps            # deptry
uv run nox -s tests           # pytest (needs extracted blobs)
```

The code is one package, `src/brother_xl2hb/`:

| Subpackage | Contents |
|---|---|
| `codec` | plane RLE encoders (Normal) and the APT codec (Fine) |
| `color` | colour grids, input remap, saturation, ordered dither |
| `protocol` | XL2HB stream writer and PJL, without model data |
| `models` | `PrinterModel` per printer: paper geometry, media and tray codes, dither set, colour grid choice, PJL, page flips |
| `render` | one page: PPM input, the per-line and the band path |
| `_native` | optional Cython kernels, each with a pure-Python fallback |

`job` (PJL, the session, page order, duplex, reverse, skip blank),
`settings`, `cli` and `cups_filter` sit at the top; `cups/brhl4150cdn-filter`
only puts the package on the path and calls `cups_filter.main`.
`scripts/extract_blobs.sh [model]` writes Brother's tables to
`src/brother_xl2hb/lut/<model>/` and `src/brother_xl2hb/color_data/<model>/`.

A printer model is one `PrinterModel` (`models/base.py`): the data its
filter writes plus a few hooks where Brother's filters differ. The
HL-4150CDN's is `models/hl4150cdn.py`. Adding a model means a module
like it, an entry in `models.MODELS`, its PPD, and the captures that
show its output is byte-identical; `tests/test_models.py` checks that
its tables cover every setting.

The Cython modules are optional in development too; build them in place
with:

```bash
uv run --with cython,setuptools python setup_cython.py build_ext --inplace
```

The test fixtures in `tests/fixtures/` are zstd-compressed XL2HB
captures from the manufacturer's filter — pytest decompresses them on
demand. Adding a new capture-based test is two steps: drop a fresh
capture into `tests/fixtures/<name>.xl2hb.zst` and reference it from a
parametrized test (`tests/test_full_pipeline.py` has the existing
pattern).

### Comparing with the manufacturer's filter

The captures cover every colour, tray and media setting, grayscale,
simplex and both duplex modes. Normal mode is compared byte for byte on
A4, Letter, A5, DL, 3x5 and Envelope #4, and on the page header values of
all 21 paper sizes; Fine mode on all 21 paper sizes.

For new comparisons, the original filter itself runs under Docker
(`--platform linux/386`). After `./scripts/extract_blobs.sh`, three
scripts in `scripts/` do the comparing:

```bash
# Render a test page or job with both drivers and diff the XL2HB streams.
uv run python scripts/fine_diff.py hramp --lines 64 --rc Brightness=-20
uv run python scripts/fine_diff.py edges --paper Letter
uv run python scripts/fine_diff.py bars --pages 4 --rc Duplex=DuplexNoTumble --rc BRReverse=ON

# Fuzz the Fine codec against the binary's own functions (Unicorn CPU emulator).
uv run --with unicorn python scripts/native_codec.py 20000 [--cython]

# Trace the original's Fine dither under gdb: its CMYK input and 4bpp output per line.
uv run python scripts/trace_dither.py solid:128,128,128 --lines 4
```

`fine_diff.py` defaults to Fine mode on A4 (`--rc BRResolution=Normal` for
Normal). It prints `IDENTICAL` or the first differing byte; for a
single-page Fine job it also decodes the original's scanlines and reports,
per plane, whether the dithered data or the encoder differs. The
original's output is cached in `build/fine-diff-cache/`, so reruns only
cost the Python side. `--save DIR` keeps both streams, `--dump DIR` the
per-line comparison. Run any script with `--help` for the test patterns
and options.

`trace_dither.py` builds a Docker image with qemu-user and gdb on first
use, as Docker Desktop's i386 emulation does not support ptrace.

## License

[GPL-3.0-or-later](LICENSE). The PPD in `cups/` is derived from
Brother's GPL-2.0-or-later HL-4150CDN PPD; see the file header for
attribution. The Brother calibration tables that `extract_blobs.sh`
pulls in remain under their own licence and never enter this
repository.

## Trademarks

Brother and HL-4150CDN are trademarks of Brother Industries, Ltd. This
project is an independent, community-driven driver and is not affiliated
with, endorsed by, or sponsored by Brother Industries, Ltd.
