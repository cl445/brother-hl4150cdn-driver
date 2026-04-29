# Brother HL-4150CDN Driver

[![CI](https://github.com/cl445/brother-hl4150cdn-driver/actions/workflows/ci.yml/badge.svg)](https://github.com/cl445/brother-hl4150cdn-driver/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](LICENSE)

Open-source CUPS driver for the Brother HL-4150CDN color laser printer.
Pure Python, runs anywhere CUPS does — including ARM (Raspberry Pi).

## Status

Drop-in for the printer's Normal mode (600 dpi). Output matches the
manufacturer's CUPS filter byte-for-byte for default settings and for
single-axis brightness/contrast/saturation/RGB-key adjustments. Three
combined-setting cases (`saturation = −20`, `contrast = −20`, and
`BRColorMatching = Vivid`) differ by a handful of dither bits in the
compressed payload — visually indistinguishable, but not byte-equal.

Fine mode (1200 dpi) emits valid framing; the compressed band data is
not yet byte-identical. Correctness is established by byte-level
comparison against captures from the manufacturer's filter.

## Requirements

- Python 3.13+ and [`uv`](https://docs.astral.sh/uv/)
- Ghostscript (the CUPS pipeline rasterizes PostScript to PPM)
- CUPS
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

## Settings

All options are PPD-driven and surfaced in the standard print dialog.
Pass them as `-o key=value` to `lp` / `lpr` for scripting.

| Option | Values | Notes |
|---|---|---|
| `PageSize` | A4, Letter, Legal, Executive, A5, JISB5, Postcard, DL, C5, Com-10, Monarch | |
| `MediaType` | Plain, Thin, Thick, Thicker, Bond, Envelope, Recycled, Label, Glossy, … | 10 entries |
| `BRResolution` | Normal, Fine | Fine mode is incomplete (see Status) |
| `BRMonoColor` | Auto, FullColor, Mono | |
| `Duplex` | None, DuplexTumble, DuplexNoTumble | Tumble = short edge |
| `BRColorMatching` | Normal, Vivid, None | Vivid not byte-identical yet |
| `TonerSaveMode` | OFF, ON | Uses the toner-save dither tables |
| `BRSkipBlank` | OFF, ON | |
| `BRReverse` | OFF, ON | Reverse page order |
| `Brightness` | −20 … +20 | |
| `Contrast` | −20 … +20 | |
| `Saturation` | −20 … +20 | |
| `RedKey`, `GreenKey`, `BlueKey` | −20 … +20 | Per-channel input shift |
| `Copies`, `Collate` | int, OFF/ON | |

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
uv run nox -s tests           # pytest (826 tests, needs extracted blobs)
```

The test fixtures in `tests/fixtures/` are zstd-compressed XL2HB
captures from the manufacturer's filter — pytest decompresses them on
demand. Adding a new capture-based test is two steps: drop a fresh
capture into `tests/fixtures/<name>.xl2hb.zst` and reference it from a
parametrized test (`tests/test_full_pipeline.py` has the existing pattern).

## License

[GPL-3.0-or-later](LICENSE). The Brother calibration tables that
`extract_blobs.sh` pulls in remain under their own licence and never
enter this repository.
