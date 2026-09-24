# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] — 2026-09-24

Output now matches the manufacturer's filter byte for byte for every RC
setting, grayscale and duplex, and on the paper sizes checked against
captures (Letter, A5, DL, 3x5, Envelope #4); all 21 sizes carry the
original's page header values. On a Raspberry Pi 3B+ a page takes about a
second instead of close to three minutes.

### Upgrading
- Re-run `scripts/extract_blobs.sh` (16 new colour grids) and
  `cups/install.sh` (Cython modules, inverse LUTs for Normal and Vivid).
- Re-assign the PPD to existing queues
  (`lpadmin -p <queue> -P /usr/share/cups/model/brhl4150cdn.ppd`): the
  duplex option is now the standard `Duplex` keyword, and the new paper
  sizes and media types are only in the new PPD. This resets the queue's
  default options.

### Fixed
- Duplex never reached the printer: the PPD option was named `BRDuplex`,
  so CUPS could not map IPP `sides` onto it. It is now the standard
  `Duplex` option (`BRDuplex` from older queues is still accepted). The
  BeginPage attributes now match the original (`DuplexPageMode` 0x00 for
  long edge, 0x81 for short edge; long-edge back pages are mirrored and
  flagged), and the whole job is one XL2HB session.
- Rows wider than the printable area (an uncropped Ghostscript render)
  crashed the dither with a broadcast error; they are now cut to width.
- Executive, Com-10 and Monarch were rasterised at Letter size unless the
  PostScript job set its own page size: Ghostscript does not know the paper
  names the CUPS filter passed. The filter now gives Ghostscript the page
  size in points for every format.
- The README's settings table listed option names the PPD does not have
  (`MediaType`, `InputSlot`, `TonerSaveMode`, `Brightness`, `RedKey`, ...);
  `lp -o` with those names was silently ignored. It now lists the PPD's
  names (`BRMediaType`, `BRInputSlot`, `BRTonerSaveMode`, `BRBrightness`,
  `BRRed`, ...) and values.
- Ghostscript's error output went to a pipe that was only read after the
  job, so a job with many Ghostscript warnings could stall the filter. It
  now goes to a temporary file.
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
- Much faster on slow ARM hosts, with byte-identical output. On a Pi 3B+
  a 131-page job through the CUPS filter takes 133 s at 64 MB peak memory;
  1.0.0 needed 167 s for a single A4 page.
  - Optional Cython modules (`_rle_fast`, `_color_fast`, `_dither_fast`,
    `_band_fast`): RLE encoding, colour lookup, dither and a band renderer
    that runs colour, dither and RLE for all planes without the GIL on up
    to three threads (`BRHL4150CDN_RENDER_THREADS` overrides). Pure Python
    remains the fallback; `install.sh` builds them when a compiler is
    available.
  - A precomputed RGB→KCMY inverse LUT (`--precompute-lut`, memory-mapped)
    replaces per-pixel interpolation.
  - Pure-white scanlines skip colour conversion and dithering.
  - The CUPS filter streams Ghostscript output in bands instead of
    holding whole pages in memory, and avoids page-sized copies.
  - `BRReverse` no longer keeps every page in memory: finished pages are
    spooled to a temporary file and written in reverse order (a 131-page
    job was OOM-killed on a Pi 3 before).
- Without an installed inverse LUT, the native band kernel interpolates the
  colour grid itself (about 1.8x slower than the inverse LUT on an M-series
  Mac, still multi-threaded) instead of the ~20x slower numpy path.
  `--precompute-lut` and `install.sh` write inverse LUTs for Normal and
  Vivid (`inverse_lut.npy`, `inverse_lut_srgb.npy`, 64 MiB each); the other
  grids always use the kernel interpolation.

- Requires numpy 2.5.3 or newer. Development tools updated (ruff 0.16,
  pyrefly 1.3, pytest 9.1); CI runs on Ubuntu 26.04 and checks
  dependencies with deptry.

- `pipeline.filter_duplex_pages` is now `filter_pages` (it renders every
  job, simplex included); the unused `lut_dir` parameters are gone.

### Removed
- `apply_vivid` (the guessed saturation boost) and `input_slot_to_tray`.
- The placeholder `main.py`.
- Code that production never reached: the M-plane 20-bit sub-block encoder
  (`encode_m_plane_20`, `rle_encode`), the `read_group` option of the RLE
  encoders, the PJL options `SOURCETRAY`, `RET`, `PAGEPROTECT` and
  `MANUALDPX` (the original never sends them for this model), the
  pattern-based pure-Python dither fallback, and bytes-only helpers that
  only tests used (`dither_cmyk_1bpp/4bpp`, `rgb_line_to_cmyk_intensities`).
- The tone-curve path (`gamma_select`, the CUPS option `BRGammaSelect` and
  the RC key `GammaSelect`, module `tone_curve`). The PPD never offered it,
  and the original filter never applies a tone curve in this mode, so it
  was the one setting whose output did not follow the original.
  `extract_blobs.sh` no longer extracts the two gamma curves.

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

[Unreleased]: https://github.com/cl445/brother-hl4150cdn-driver/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/cl445/brother-hl4150cdn-driver/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/cl445/brother-hl4150cdn-driver/releases/tag/v1.0.0
