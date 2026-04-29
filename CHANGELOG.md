# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-04-28

First public release. Functionally complete drop-in for the HL-4150CDN's
Normal mode (600 dpi).

### Added
- Full PPM → XL2HB pipeline: 3D-LUT colour separation, ordered
  dithering, per-plane RLE compression.
- CUPS integration: filter (`cups/brhl4150cdn-filter`) and PPD with all
  settings exposed (paper, duplex, brightness, contrast, saturation,
  per-channel RGB shifts, vivid/none colour matching, toner save,
  blank-page skip, reverse output).
- `scripts/extract_blobs.sh`: pulls the official Brother LPR driver
  `.deb`, MD5-verifies it, and extracts the printer-calibration tables
  into `src/lut/` and `src/color_data/`. The blobs stay in the local
  working tree and never enter git history.
- 828 tests covering compression, dithering, colour separation, PJL,
  XL2HB framing, and 14 byte-identity captures from the manufacturer's
  filter under varied RC settings.
- Fine mode (1200 dpi) framing layer including 4bpp dithering and the
  toner-save Fine BRCD path.

### Verified byte-identical against the manufacturer's filter
- All-white, all-black, K-only fullwidth/halfpage/narrow PPMs.
- Single-band colour pages: cyan, gray (50% and 75%), red.
- Single-setting variants on a cyan band: baseline, saturation +20,
  green +20, blue +20, BRColorMatching=None, brightness ±20,
  contrast +20, red +20, combined (B/C/S=10), toner-save.

### Known limitations
- Three setting variants drift by a handful of dither bits in the
  compressed payload (visually identical, not byte-equal): saturation
  −20 (21 bytes diff), contrast −20 (88 bytes diff), and
  BRColorMatching=Vivid (different colour path).
- Fine mode emits valid framing but its compression codec is not yet
  byte-identical with the manufacturer's filter.
