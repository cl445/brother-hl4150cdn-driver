# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- The CUPS filter now applies the queue's PPD defaults (`$PPD`) before the
  job's options, as Brother's cupswrapper does. CUPS passes only the
  options a job carries, and jobs from desktop clients carry no `BR*`
  options, so defaults such as `DefaultBRGray: ON` had no effect. An IPP
  `sides` in the job still overrides the PPD's default duplex.
- Grayscale mode (`BRMonoColor=Mono`) only switched the PJL header to
  `GRAYSCALE` and still sent CMY planes. It now matches the manufacturer's
  filter byte for byte: the BeginImage band config clears its colour flag,
  and K is `255 - luma` (Rec. 601, integer rounding as in
  `compress_separate_mono`) with C/M/Y left empty. Saturation, vivid and
  improve-gray have no effect in this mode, the same as in the original.
- The M plane's 10-bit RLE encoder differed from `compress_encode_plane_m`
  around context skips: after a run it jumped straight into a skip, and it
  emitted the word that starts a context-predicted stretch as `run(1)` on
  top of counting it in the skip. Busy colour pages now match byte for byte.
  This was the cause of the contrast −20 and saturation −20 capture drift.
- Colour separation now uses the grid the original picks in
  `lookup_color_transform_table`, out of 18: profile rgb (Normal), srgb
  (Vivid) or cmyk (colour matching None); the ImpGray variant for
  `BRGray=ON`; density2 for toner save; glossy for glossy media.
  `BREnhanceBlkPrt` prints pure black as rich black (grid entry 0). Before,
  Vivid used a guessed saturation boost, None a plain RGB→CMY split, and
  toner save, glossy, BRGray and BREnhanceBlkPrt did not change the colours.
  `scripts/extract_blobs.sh` extracts the 16 new grids; re-run it and
  `install.sh` on existing installs.
- Colour matching None ignores brightness, contrast, RGB keys and
  saturation, as the original does (`cmyk_basic`).
- `BRGray` only writes `@PJL SET IMPROVEGRAY=ON` and `BREnhanceBlkPrt` writes
  `@PJL SET UCRGCRFORIMAGE=ON`. Before, `BRGray` wrote both lines and
  `BREnhanceBlkPrt` was ignored.
- `BRMonoColor=FullColor` writes `COLORADAPT=OFF`; only `Auto` sets it `ON`.
- The input tray goes into the BeginPage MediaSource attribute (Auto 1,
  Tray1 1001, Tray2 1005, MP tray 1004, manual 2), as in the original.
  Before, it was always 1, and Tray1/Tray2 added a `@PJL SET SOURCETRAY` line
  the original never sends.
- Media type names on the wire match the original (`Thick2`, `Envelopes`,
  `Envthick`, Bond as `Regular`). The media types EnvThin and Postcard were
  added, and Brother's own spellings (`BOND`, `Env`, `PostCard`) are accepted.
- Paper sizes other than A4 match the original: the source height follows
  the original's band height, `min(ImagingArea height, paper height) in
  whole points * 600 // 72` (e.g. Letter 6400 rows, not 6396), and the
  MediaSize codes are fixed (A5, JISB5, EnvDL, Env10, EnvMonarch were
  wrong). The ten sizes the original supports beyond those (A6, ISO B5/B6,
  JIS B6, A5 long edge, 3x5, Folio, DL long edge, Envelope #4/MAX) were
  added; several are sent as MediaSize name strings, as the original does.
- K and Y lines whose width is not a multiple of 12 bits lost ink in their
  last pixels: the encoder dropped the trailing bits instead of zero-padding
  the last 12-bit word (`read_word_16`). This only hit sizes without right
  padding (A5, A6, Env10, 3x5), never A4.
- Positive saturation now truncates the mid channel like the binary (FPU
  control word 0xC at 0x0804f846), including its 80-bit `(a / b) * c`
  evaluation, instead of rounding half up.

### Changed
- Without an installed inverse LUT, the native band kernel interpolates the
  colour grid itself (about 1.8x slower than the inverse LUT on an M-series
  Mac, still multi-threaded) instead of the ~20x slower numpy path.
  `--precompute-lut` and `install.sh` write inverse LUTs for Normal and
  Vivid (`inverse_lut.npy`, `inverse_lut_srgb.npy`, 64 MiB each); the other
  grids always use the kernel interpolation.

### Removed
- `apply_vivid` (the guessed saturation boost) and `input_slot_to_tray`.

### Known limitations
- Fine mode (1200 dpi) is still incomplete (APT compression, Phase 3).

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
