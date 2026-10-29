# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The code is one package, `brother_xl2hb`, with the subpackages `codec`,
  `color`, `protocol`, `render` and `_native` (Cython). The CLI without
  CUPS is `uv run brother-xl2hb`; `src/brfilter.py` and the
  `brother_encode` re-export module are gone.
- `protocol.xl2hb` holds only the XL2HB protocol; the HL-4150CDN's paper
  sizes, media codes and tray values are in `models.hl4150cdn`, PJL in
  `protocol.pjl`. `render.pipeline` is split into `job` (session, page
  order, duplex and reverse flips, skip blank) and `render.rows`,
  `render.lines`, `render.bands` and `render.page`.
- Everything model-specific is one `PrinterModel`, `models.HL4150CDN`:
  paper sizes and points, page origin, media and tray codes, the dither
  table set, the colour grid choice, PJL and which pages are sent
  bottom-up. `job.filter_pages`, the CUPS filter and the CLI take a model
  and default to the HL-4150CDN; the CLI has `--model`.
- Settings are read through the model (`settings_from_rc_file`,
  `settings_from_cups_options`): Brother's spellings (Env, A5Rotated,
  B5, B6) are the model's `aliases`, and a choice the model does not
  support falls back to the default with a warning.
- The HL-4150CDN's tables are reached through `models.HL4150CDN` only;
  the module constants of `models/hl4150cdn.py` are private.
- `cups/install.sh` and `cups/uninstall.sh` take `--model` (default
  hl4150cdn); paths, filter, PPD and queue names follow from it. The
  CUPS log prefix comes from the model, and the render threads are set
  with `BRXL2HB_RENDER_THREADS` (`BRHL4150CDN_RENDER_THREADS` still works).
- Only models that declare `generated_grids` fall back to the generated
  colour grids when their blobs are missing; for any other model a
  missing grid is an error that names `extract_blobs.sh <model>`.
- The model decides how each page is sent (`page_sends`: bottom-up or
  not, PageCopies, and how often); the job no longer knows about duplex
  sides.
- `scripts/extract_blobs.sh` puts Brother's tables into
  `src/brother_xl2hb/lut/<model>/` and `src/brother_xl2hb/color_data/<model>/`
  (the unpacked package to `.brother-blobs/<model>/`); re-run it
  after pulling. `cups/install.sh` replaces the flat module layout of
  1.2.0 in `/usr/local/lib/brhl4150cdn/`.
- `scripts/extract_blobs.sh` checks the downloaded package's SHA-256
  instead of its MD5.
- `cups/install.sh` installs numpy below 3 and builds the native modules
  with Cython 3.3.0 and setuptools 84.2.1, the versions CI uses. It
  warns about missing colour grids as well as missing dither tables.
- `scripts/fine_diff.py` and `scripts/trace_dither.py` cache in `.cache/`
  instead of `build/`, which is left to setuptools.
- CI also runs the tests on Python 3.14.

### Fixed
- The CUPS filter imports the package installed beside it first; a
  `src/` directory next to its own directory (the repository layout) is
  only a fallback when that package is missing.
- Literal stretches longer than one block are encoded as in the original:
  the K/Y encoder flushes at 4095 words instead of 2047, and no encoder
  repeats the last word after the flush. A scanline has at most about 425
  words, so printed output does not change (#3).
- Grayscale (BRMonoColor=Mono) ignores the Red, Green and Blue keys and
  keeps Brightness and Contrast with BRColorMatching=None, as the
  original does.

## [1.2.0] — 2026-10-28

Fine mode (600x2400 dpi) matches the manufacturer's filter byte for byte
and can be chosen in the print dialog. Reverse print, copies and skip
blank pages in duplex now behave as with the original driver. Jobs that
hand whole pages to the renderer hold at most one page at a time, which
halves the CUPS filter's peak memory with skip blank.

### Upgrading
- Re-run `cups/install.sh`: it builds the new Cython module for Fine
  mode (`_fine_fast`); without the module, Fine pages are rendered in
  pure Python, about 30 times slower.
- Re-assign the PPD to existing queues to get the Fine choice into the
  print dialog (see Upgrading in the README).

### Fixed
- Reverse print (`BRReverse=ON`) keeps the page order and sends every page
  bottom-up with the page marker the manufacturer's filter uses for it;
  before, it reversed the page order. On the printer the pages come out
  turned by 180 degrees in their original order.
  Long-edge duplex jobs therefore no longer need the page count up front:
  the CUPS filter streams them from stdin like any other job instead of
  spooling them and counting their pages, and `filter_pages` has lost its
  `page_count` argument.
- Skip blank pages (`BRSkipBlank=ON`) also works in duplex; the next page
  takes the blank one's side, as in the original.
- Bond paper is sent as media type `Transparency`, which is what the
  original sends for the `MediaType=BOND` that Brother's CUPS wrapper
  writes. The earlier `Regular` came from a probe with the wrong rc
  spelling.
- Jobs that hand whole pages to the renderer held up to three A4 rasters
  (~100 MB each) at once: the page peeked to detect an empty job stayed
  referenced by `itertools.chain` for the whole job, and `enumerate` and
  the blank-page filter kept the previous page while the next was read.
  The CUPS filter with skip blank now peaks at about 170 MB instead of
  340 MB (Linux, A4), the CLI at about 230 MB instead of 320 MB.
- Negative brightness shifted one step too little: the shift is
  `brightness * 255 >> 7`, which rounds down, not toward zero
  (Brightness=-20 shifts by 40, not 39). Affects Normal mode too.
- Copies (`lp -n 2`) printed each page copies² times, so four sheets per
  page for two copies: Ghostscript already renders every page `copies`
  times, because pstops sets `NumCopies`, and the filter asked the
  printer for that many copies of each page again. The CUPS filter now
  sends one copy per rendered page, like Brother's cupswrapper, which
  passes its filter `Copies=1`.

### Changed
- Fine mode (600x2400 dpi) is byte-identical to the original filter for
  all 21 paper sizes and in duplex: the Fine codec, the 4bpp dither
  tables and the original's dither overrun are reproduced, and the image
  width is rounded up to 8 pixels as in the original. A native band
  kernel renders an A4 Fine page in well under a second on an M-series
  Mac and in about 6.5 s on a Raspberry Pi 3B+. The PPD offers it again
  as "Fine (2400 dpi class)" and the CUPS filter prints it instead of
  falling back to Normal. Checked on the printer.
- `scripts/fine_diff.py` compares any job with the original filter in
  Docker, writing the rc for the original with Brother's spellings
  (`BOND`, `Env`); `scripts/trace_dither.py` traces the original's Fine
  dither under gdb.

## [1.1.1] — 2026-09-26

### Upgrading
- Re-run `cups/install.sh`: it now precomputes four inverse LUTs (Normal
  and Vivid, each with and without improve gray; about 260 MB, four
  minutes on a Pi 3) and removes the old ones.
- Re-assign the PPD to existing queues if you want the Fine choice gone
  from the print dialog; the filter prints Fine requests in Normal anyway.

### Fixed
- Jobs with the queue's default improve gray (`BRGray: ON`) had no
  precomputed inverse LUT and were interpolated per pixel. With the new
  LUTs a page takes about a third less CPU time on a Pi 3B+; the output
  is unchanged.
- If Ghostscript failed in the middle of a job, the printer got a half
  page and an unterminated session, Ghostscript's messages never reached
  the CUPS log and the process was not reaped. Pages are now written
  only once complete; on an error the session and the PJL footer are
  still closed, Ghostscript is stopped and its stderr is logged as
  `ERROR:`, and the filter exits with status 1.
- An inverse LUT computed from the generated fallback grids (e.g.
  `install.sh` before `extract_blobs.sh`) kept overriding the real grids.
  LUT file names now carry a hash of the grids they come from; others are
  ignored with a warning.
- `count_ps_pages` could take `%%Pages: 1` of an embedded EPS for the job's
  page count when the header said `(atend)`.
- A non-numeric `BRHL4150CDN_RENDER_THREADS` crashed the filter; it now
  falls back to the default with a warning.
- The CLI read only the first page from stdin, ignored a missing `--rc`
  file, and `--precompute-lut` wrote the LUTs under names the filter does
  not read.

### Changed
- Fine mode (600x2400 dpi) is no longer offered in the PPD, as it has not
  been verified on the printer. A Fine request is printed in Normal mode
  with a `WARNING:` in the CUPS log.
- Missing dither tables (Bayer fallback) are now logged as a warning.
- `install.sh` and `uninstall.sh` refuse to run on macOS, where the CUPS
  directories are SIP-protected; that branch never worked.
- The CUPS filter is now checked by ruff and pyrefly, and `main()` is
  tested end to end with a fake Ghostscript.

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
  `Envthick`). The media types EnvThin and Postcard were
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

[Unreleased]: https://github.com/cl445/brother-hl4150cdn-driver/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/cl445/brother-hl4150cdn-driver/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/cl445/brother-hl4150cdn-driver/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/cl445/brother-hl4150cdn-driver/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/cl445/brother-hl4150cdn-driver/releases/tag/v1.0.0
