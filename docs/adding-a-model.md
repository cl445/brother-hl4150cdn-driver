# Adding a printer model

This driver is built so that a second Brother model is mostly data: one
`PrinterModel`, its PPD, a few script entries and captures from Brother's
own filter that prove the output is identical. Check
[Other Brother printers](other-printers.md) first: it says which group your
model is in and what is already known to differ.

You need the printer for the last step only. Everything before it runs
against Brother's filter in Docker.

## What you need

- Linux or macOS with Docker (Brother's filter is a 32-bit x86 program and
  runs in an `i386/debian:bookworm-slim` container, on ARM too, through
  emulation)
- [`uv`](https://docs.astral.sh/uv/) and a checkout of this repository
  (`uv sync --all-extras`, see the README's Development section)
- Brother's LPR package (`<model>lpr-<version>.i386.deb`) and cupswrapper
  package for your model. Brother's index
  `https://download.brother.com/pub/com/linux/linux/infs/<MODEL>` (for
  example `.../infs/MFC-9970CDW`) names them; the files are under
  `https://download.brother.com/pub/com/linux/linux/packages/`.

## 1. Get Brother's tables

`scripts/extract_blobs.sh` downloads the LPR package, checks its checksum
and copies the tables into `src/brother_xl2hb/lut/<model>/` (dither tables)
and `src/brother_xl2hb/color_data/<model>/` (colour grids and
interpolation tables). Add a `case` block for your model with:

- the package URL and its checksum
- the filter's path inside the package and the `inf/lut` directory
- the offsets of the colour grids and interpolation tables inside the
  filter binary. They are not separate files. The colour matching *None*
  grids (`cmyk_*`) and the interpolation tables are the same in every
  XL2HB family, so searching the binary for the HL-4150CDN's copies
  finds them and shows where the others start.

The tables are Brother's and never go into this repository; only the
script that fetches them does.

## 2. Describe the model

Create `src/brother_xl2hb/models/<model>.py` after
`models/hl4150cdn.py` and register it in `models.MODELS`. The fields are
documented in `models/base.py`; most come straight from Brother's
package:

| Field | Where it comes from |
|---|---|
| `paper_sizes` | `inf/paperinfij2` (pixels at 600 dpi) |
| `imaging_height_pt` | `inf/ImagingArea` |
| `paper_points` | the PPD's `PaperDimension` lines |
| `media_types`, `media_source`, `duplex_page_mode`, `media_size` | the values Brother's filter writes into BeginPage; read them from a capture |
| `dither_cache` | the suffix of the `0600-*_cacheNN.bin` files |
| `pjl_header`, `pjl_footer` | the PJL around the XL2HB stream in a capture |
| `color_table`, `page_sends` | hooks; start from the HL-4150CDN's and change them where captures disagree |
| `aliases` | spellings in Brother's rc file and PPD that differ from this driver's |

Set `generated_grids=False`: the generated grids are fitted to the
HL-4150CDN.

## 3. Package it for CUPS

- `cups/br<model>.ppd`, derived from Brother's PPD in the cupswrapper
  package (keep its GPL header and attribution)
- `cups/br<model>-filter`, a copy of `cups/brhl4150cdn-filter` that calls
  `main(MODELS["<model>"])`
- a `case` block for the model in `cups/install.sh` and
  `cups/uninstall.sh`

`tests/test_model_packaging.py` checks all of this against the model's
tables and fails until everything agrees.

## 4. Prove the output is identical

Run Brother's filter and this driver on the same pages and compare the
XL2HB streams byte for byte. The README's section *Comparing with the
manufacturer's filter* shows how the HL-4150CDN's filter runs under
Docker; `scripts/fine_diff.py` automates it but is wired to the
HL-4150CDN so far, so a `--model` option is a good first change.

Cover at least: Normal and Fine, colour and grayscale, simplex and
duplex, a few paper sizes and colour settings. Store each capture of
Brother's filter as `tests/fixtures/<name>.xl2hb.zst` (zstd) and add a
test that renders the same page and compares it with the capture, as
`tests/test_full_pipeline.py` and `tests/test_fine_mode.py` do. Fixtures
always come from Brother's filter, never from this driver.

If a model needs behaviour the HL-4150CDN does not have (other Normal-mode
encoders, a different colour stage), add it as a field or hook on
`PrinterModel` in the same pull request, with the captures that show it.

## 5. Print

Install with `sudo cups/install.sh --model <model>` and print colour,
grayscale, duplex and Fine pages. Mention in the pull request what you
printed and on which machine.

## Pull request

Open a pull request against `main`. CI runs the tests with and without the
Cython modules. It runs `scripts/extract_blobs.sh` for the HL-4150CDN
only, so add a call for your model to `.github/workflows/ci.yml`; the
block must work unattended. Reviews look mostly at the
captures and the tests: if they show identical output, the rest follows.
