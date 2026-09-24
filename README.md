# Brother HL-4150CDN Driver

[![CI](https://github.com/cl445/brother-hl4150cdn-driver/actions/workflows/ci.yml/badge.svg)](https://github.com/cl445/brother-hl4150cdn-driver/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

Open-source CUPS driver for the Brother HL-4150CDN color laser printer.
Written in Python with optional Cython acceleration, runs anywhere CUPS
does — including ARM (Raspberry Pi).

## Status

Drop-in for the printer's Normal mode (600 dpi). Output matches the
manufacturer's CUPS filter byte for byte for every colour, tray and media
setting (colour matching Normal/Vivid/None, improve gray, enhance black,
toner save, glossy, brightness/contrast/saturation/RGB keys), for
grayscale, and for simplex, long-edge and short-edge duplex. Paper sizes
other than A4 are checked byte for byte on Letter, A5, DL, 3x5 and
Envelope #4; for all 21 sizes the page header values match the original.
Correctness is established by byte-level comparison against captures from
the manufacturer's filter.

Fine mode (1200 dpi) emits valid framing; the compressed band data is
not yet byte-identical.

On a Raspberry Pi 3B+ with the Cython modules and the precomputed
inverse LUTs, a 131-page job goes through the CUPS filter in about
133 s (roughly 1 s per page) with a peak memory use of about 64 MB.

## Requirements

- Python 3.13+ and [`uv`](https://docs.astral.sh/uv/)
- Ghostscript (the CUPS pipeline rasterizes PostScript to PPM)
- CUPS
- Optional, for speed: a C compiler and the Python headers (Debian:
  `sudo apt install build-essential python3-dev`). `install.sh` then
  builds the Cython modules (RLE encoders, colour lookup, dither and a
  band renderer that uses several cores); without them the driver falls
  back to pure Python, which is several times slower.
- About 130 MB of disk space for the precomputed inverse colour LUTs
  (Normal and Vivid, 64 MiB each). `install.sh` computes them, which takes
  about two minutes on a Pi 3.
- A copy of the official Brother HL-4150CDN LPR driver `.deb`
  (`scripts/extract_blobs.sh` downloads and verifies it; the printer's
  calibration tables are extracted into `src/lut/` and `src/color_data/`
  and are not redistributed)

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
(e.g. the standard `Duplex` option since 1.1.0), re-assign it. This resets
the queue's default options:

```bash
sudo lpadmin -p Brother_HL-4150CDN -P /usr/share/cups/model/brhl4150cdn.ppd
```

## Settings

All options are PPD-driven and surfaced in the standard print dialog.
Pass them as `-o key=value` to `lp` / `lpr` for scripting.

| Option | Values | Notes |
|---|---|---|
| `PageSize` | A4, Letter, Legal, Executive, A5, PRA5Rotated (A5 long edge), A6, ISOB5, ISOB6, JISB5, JISB6, Postcard, EnvDL, EnvPRC5Rotated (DL long edge), EnvC5, Env10 (Com-10), EnvMonarch, Br3x5, FanFoldGermanLegal (Folio), EnvYou4 (Envelope #4), EnvChou3 (Envelope MAX) | |
| `BRMediaType` | Plain, Thin, Thick, Thicker, Bond, Envelope, EnvThin, EnvThick, Recycled, Postcard, Label, Glossy | Glossy uses the glossy colour tables |
| `BRResolution` | 600dpi (Normal), 600x2400dpi (Fine) | Fine mode is incomplete (see Status) |
| `BRMonoColor` | Auto, FullColor, Mono | |
| `Duplex` | None, DuplexNoTumble, DuplexTumble | Tumble = short edge; IPP `sides` works too |
| `BRColorMatching` | Normal, Vivid, None | Each selects its own colour tables |
| `BRGray` | OFF, ON | Improve gray: ImpGray colour tables |
| `BREnhanceBlkPrt` | OFF, ON | Enhance black: rich black for pure black |
| `BRImproveOutput` | OFF, BRLessPaperCurl, BRFixIntensity | |
| `BRInputSlot` | AutoSelect, Tray1, Tray2, MPTray | |
| `BRTonerSaveMode` | OFF, ON | Toner-save dither and colour tables |
| `BRSkipBlank` | OFF, ON | |
| `BRReverse` | OFF, ON | Reverse page order |
| `BRBrightness` | -20 … 20 | |
| `BRContrast` | -20 … 20 | |
| `BRSaturation` | -20 … 20 | |
| `BRRed`, `BRGreen`, `BRBlue` | -20 … 20 | Per-channel input shift |

Copies come from the standard `-n` / `copies` option.

## CLI without CUPS

```bash
# PPM in, XL2HB raw stream out
cat page.ppm | uv run python src/brfilter.py \
    --paper Letter --duplex long --toner-save \
  > page.xl2hb

# Send straight to the printer (network)
cat page.ppm | uv run python src/brfilter.py | nc <printer-ip> 9100

# …or via USB
cat page.ppm | uv run python src/brfilter.py > /dev/usb/lp0
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
uv run nox -s tests           # pytest (1109 tests, needs extracted blobs)
```

The Cython modules are optional in development too; build them in place
with:

```bash
uv run --with cython,setuptools python setup_cython.py build_ext --inplace
```

The test fixtures in `tests/fixtures/` are zstd-compressed XL2HB
captures from the manufacturer's filter — pytest decompresses them on
demand. Adding a new capture-based test is two steps: drop a fresh
capture into `tests/fixtures/<name>.xl2hb.zst` and reference it from a
parametrized test (`tests/test_full_pipeline.py` has the existing pattern).

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
