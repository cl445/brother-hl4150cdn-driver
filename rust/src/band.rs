//! Render a band of RGB scanlines to encoded 1bpp planes (Normal mode).
//!
//! Colour lookup, ordered dither and sliding-window RLE for a whole band in
//! one call without the GIL, so bands can render on several threads. Output
//! is byte-identical to the per-line path in lines.encode_lines: colour
//! through the inverse LUT (or interpolated in the colour grid), 1bpp
//! threshold dither, then encode_plane_12bit / encode_plane_20bit /
//! encode_plane_10bit. `render_band_mono` does the same for grayscale pages,
//! matching lines.encode_lines_mono. Port of `_band_fast.pyx`.
//!
//! Each row takes two passes: the ink values of every pixel (looked up once
//! per run of equal pixels, and not at all for a row equal to the last
//! converted one), then the threshold dither per plane. The ink values are
//! kept per plane, so the dither runs over contiguous bytes, which the
//! compiler vectorizes.

use crate::color::Grid;
use crate::rle::LineEncoder;

/// RLE word size per plane (K, C, M, Y), as in plane_encoders.
const BITS: [u32; 4] = [12, 20, 10, 12];

/// Entries of the flat KCMY inverse LUT: 4 bytes for every RGB colour.
pub const LUT_SIZE: usize = 256 * 256 * 256 * 4;

/// Where a pixel's K, C, M, Y come from.
#[derive(Clone, Copy)]
pub enum Colour<'a> {
    /// The flat inverse LUT, LUT_SIZE bytes.
    Lut(&'a [u8]),
    /// Interpolation in the colour grid.
    Grid(Grid<'a>),
}

/// A threshold matrix tiled to at least the image width, row-major.
#[derive(Clone, Copy)]
pub struct Thresholds<'a> {
    pub data: &'a [u8],
    pub rows: usize,
    pub cols: usize,
}

impl<'a> Thresholds<'a> {
    /// The first `n` thresholds of the row for page line `line`.
    fn row(&self, line: usize, n: usize) -> &'a [u8] {
        &self.data[(line % self.rows) * self.cols..][..n]
    }
}

/// One colour band: the inputs `render_band` checked.
pub struct Band<'a> {
    /// RGB scanlines, each at least `width * 3` bytes.
    pub rows: &'a [&'a [u8]],
    /// Per row, a set flag emits a blank line without looking at it; None
    /// skips the rows that are pure white over `width`.
    pub skip: Option<&'a [u8]>,
    /// Pixels per input row.
    pub width: usize,
    /// Page line index of rows[0]; selects the threshold rows.
    pub first_line: usize,
    /// Image width in pixels; pixels past `width` count as white.
    pub sw: usize,
    pub colour: Colour<'a>,
    /// K, C, M, Y threshold matrices, at least `min(width, sw)` columns.
    pub thresholds: [Thresholds<'a>; 4],
}

/// One grayscale band: the inputs `render_band_mono` checked.
pub struct MonoBand<'a> {
    /// RGB scanlines, each at least `min(width, sw) * 3` bytes.
    pub rows: &'a [&'a [u8]],
    pub width: usize,
    pub first_line: usize,
    pub sw: usize,
    /// R, G and B input remap tables, or None.
    pub remap: Option<&'a [u8; 768]>,
    /// K threshold matrix, at least `min(width, sw)` columns.
    pub thresholds: Thresholds<'a>,
}

fn all_white(bytes: &[u8]) -> bool {
    bytes.iter().fold(0xFF, |acc, &b| acc & b) == 0xFF
}

/// Dots of up to 8 values: bit 7 - k set where ink (255 - v[k]) exceeds t[k].
#[inline(always)]
fn dither8(v: &[u8], t: &[u8]) -> u8 {
    let mut b = 0u8;
    for (k, (&v, &t)) in v.iter().zip(t).enumerate() {
        b |= u8::from(!v > t) << (7 - k);
    }
    b
}

/// Dots of 16 values as 2 bytes, MSB first.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dither16(v: &[u8; 16], t: &[u8; 16]) -> [u8; 2] {
    use std::arch::aarch64::*;
    const WEIGHTS: [u8; 16] = [128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1];
    // SAFETY: NEON is part of every aarch64 target; the loads read 16 bytes
    // from 16-byte arrays.
    unsafe {
        let ink = vmvnq_u8(vld1q_u8(v.as_ptr()));
        let dots = vandq_u8(
            vcgtq_u8(ink, vld1q_u8(t.as_ptr())),
            vld1q_u8(WEIGHTS.as_ptr()),
        );
        // Three pairwise adds sum each group of 8 weighted bits into one byte.
        let m = vpaddq_u8(dots, dots);
        let m = vpaddq_u8(m, m);
        let m = vpaddq_u8(m, m);
        [vgetq_lane_u8::<0>(m), vgetq_lane_u8::<1>(m)]
    }
}

/// Dots of 16 values as 2 bytes, MSB first.
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn dither16(v: &[u8; 16], t: &[u8; 16]) -> [u8; 2] {
    [dither8(&v[..8], &t[..8]), dither8(&v[8..], &t[8..])]
}

/// Set a dot where ink (255 - value) exceeds the threshold, MSB first, into
/// out[..values.len().div_ceil(8)]. Returns whether any dot is set.
fn dither_plane(values: &[u8], thresholds: &[u8], out: &mut [u8]) -> bool {
    let thresholds = &thresholds[..values.len()];
    let (v16, v_rest) = values.as_chunks::<16>();
    let (t16, t_rest) = thresholds.as_chunks::<16>();
    let (head, tail) = out.split_at_mut(2 * v16.len());
    let mut any = 0u8;
    for ((o, v), t) in head.as_chunks_mut::<2>().0.iter_mut().zip(v16).zip(t16) {
        *o = dither16(v, t);
        any |= o[0] | o[1];
    }
    // The last 0 to 15 values, 8 per byte.
    for ((o, v), t) in tail.iter_mut().zip(v_rest.chunks(8)).zip(t_rest.chunks(8)) {
        *o = dither8(v, t);
        any |= *o;
    }
    any != 0
}

/// Packed R << 16 | G << 8 | B.
#[inline(always)]
fn rgb(px: &[u8; 3]) -> u32 {
    (u32::from(px[0]) << 16) | (u32::from(px[1]) << 8) | u32::from(px[2])
}

/// Pass 1: the K, C, M, Y values of the first `planes[0].len()` pixels of
/// `row` into `planes`. Returns whether any value has ink (is below 255).
#[inline(always)]
fn convert_row(row: &[u8], ink: impl Fn(u32) -> [u8; 4], planes: &mut [Vec<u8>; 4]) -> bool {
    let [k, c, m, y] = planes;
    let npx = k.len();
    let pixels = &row.as_chunks::<3>().0[..npx];
    let mut prev = u32::MAX;
    let mut value = [255u8; 4];
    let mut no_ink = [0xFFu8; 4];
    for (j, px) in pixels.iter().enumerate() {
        let packed = rgb(px);
        if packed != prev {
            prev = packed;
            value = ink(packed);
            for p in 0..4 {
                no_ink[p] &= value[p];
            }
        }
        k[j] = value[0];
        c[j] = value[1];
        m[j] = value[2];
        y[j] = value[3];
    }
    no_ink != [0xFF; 4]
}

/// `render_band` for one colour source.
#[inline(always)]
fn render_rows(
    band: &Band<'_>,
    ink: impl Fn(u32) -> [u8; 4] + Copy,
    lengths: &mut [i32],
) -> Vec<u8> {
    let npx = band.width.min(band.sw);
    let bpl = band.sw.div_ceil(8);
    let mut values: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; npx]);
    // Bytes past the image stay zero: the dither writes only the first ceil(npx / 8).
    let mut planes: [Vec<u8>; 4] = std::array::from_fn(|_| vec![0u8; bpl]);
    let mut prev_row: Option<&[u8]> = None;
    let mut has_ink = false;
    let mut encoder = LineEncoder::default();
    let mut out = Vec::new();
    for (i, (&row, lens)) in band
        .rows
        .iter()
        .zip(lengths.as_chunks_mut::<4>().0)
        .enumerate()
    {
        *lens = [0; 4];
        let blank = match band.skip {
            Some(skip) => skip[i] != 0,
            None => all_white(&row[..band.width * 3]),
        };
        if blank {
            continue;
        }
        let pixels = &row[..npx * 3];
        if prev_row != Some(pixels) {
            has_ink = convert_row(row, ink, &mut values);
            prev_row = Some(pixels);
        }
        if !has_ink {
            continue;
        }
        let line = band.first_line + i;
        for (pid, len) in lens.iter_mut().enumerate() {
            let thresholds = band.thresholds[pid].row(line, npx);
            if dither_plane(&values[pid], thresholds, &mut planes[pid]) {
                *len = encoder.encode(&planes[pid], BITS[pid], &mut out) as i32;
            }
        }
    }
    out
}

/// Colour-convert, dither and encode the band.
///
/// `lengths` gets 4 entries per row: the encoded length per plane (K, C,
/// M, Y), 0 when the plane line has no ink. Returns the encoded plane
/// lines back to back, in line then plane order.
pub fn render_band(band: &Band<'_>, lengths: &mut [i32]) -> Vec<u8> {
    match band.colour {
        Colour::Lut(lut) => {
            // One 4-byte entry per RGB colour; a 24-bit index never leaves the table.
            let (entries, _) = lut.as_chunks::<4>();
            let entries: &[[u8; 4]; 1 << 24] = entries.try_into().expect("LUT_SIZE bytes");
            render_rows(band, |px| entries[(px & 0xFF_FFFF) as usize], lengths)
        }
        Colour::Grid(grid) => {
            let ink = |px: u32| grid.interp_pixel((px >> 16) as u8, (px >> 8) as u8, px as u8);
            render_rows(band, ink, lengths)
        }
    }
}

/// Convert the band to grayscale, dither and encode the K plane.
///
/// K ink is 255 - luma, luma = (299 R + 587 G + 114 B + 499) // 1000 after
/// the optional input remap; pure white is not remapped. `lengths` gets
/// the encoded K length per row, 0 for no ink. Returns the encoded K lines
/// back to back.
pub fn render_band_mono(band: &MonoBand<'_>, lengths: &mut [i32]) -> Vec<u8> {
    let npx = band.width.min(band.sw);
    let bpl = band.sw.div_ceil(8);
    let mut luma = vec![0u8; npx];
    let mut plane = vec![0u8; bpl];
    let mut prev_row: Option<&[u8]> = None;
    let mut has_ink = false;
    let mut encoder = LineEncoder::default();
    let mut out = Vec::new();
    for (i, (&row, len)) in band.rows.iter().zip(lengths.iter_mut()).enumerate() {
        *len = 0;
        let pixels = &row[..npx * 3];
        if all_white(pixels) {
            continue;
        }
        if prev_row != Some(pixels) {
            let mut prev = u32::MAX;
            let mut level = 255u8;
            has_ink = false;
            for (j, px) in pixels.as_chunks::<3>().0.iter().enumerate() {
                let packed = rgb(px);
                if packed != prev {
                    prev = packed;
                    let [mut r, mut g, mut b] = px.map(u32::from);
                    if let Some(remap) = band.remap
                        && packed != 0xFF_FFFF
                    {
                        r = u32::from(remap[r as usize]);
                        g = u32::from(remap[256 + g as usize]);
                        b = u32::from(remap[512 + b as usize]);
                    }
                    level = ((299 * r + 587 * g + 114 * b + 499) / 1000) as u8;
                    has_ink |= level != 255;
                }
                luma[j] = level;
            }
            prev_row = Some(pixels);
        }
        if !has_ink {
            continue;
        }
        let thresholds = band.thresholds.row(band.first_line + i, npx);
        if dither_plane(&luma, thresholds, &mut plane) {
            *len = encoder.encode(&plane, 12, &mut out) as i32;
        }
    }
    out
}

#[cfg(feature = "python")]
pub use bindings::register;

#[cfg(feature = "python")]
mod bindings {
    use pyo3::buffer::PyBuffer;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use super::{Band, Colour, LUT_SIZE, MonoBand, Thresholds};
    use crate::buffer::{byte_rows, contiguous, contiguous_mut, matrix, matrix_mut};
    use crate::color::{GRID_ENTRIES, Grid, WEIGHT_BYTES};

    fn err<T>(msg: &str) -> PyResult<T> {
        Err(PyValueError::new_err(msg.to_owned()))
    }

    fn thresholds(buf: &PyBuffer<u8>) -> PyResult<Thresholds<'_>> {
        let (data, rows, cols) = matrix(buf)?;
        Ok(Thresholds { data, rows, cols })
    }

    /// Check the shared geometry arguments; returns (width, first_line, sw).
    fn geometry(width: isize, first_line: isize, sw: isize) -> PyResult<(usize, usize, usize)> {
        if sw <= 0 {
            return err("sw must be positive");
        }
        if width < 0 || first_line < 0 {
            return err("width and first_line must not be negative");
        }
        Ok((width as usize, first_line as usize, sw as usize))
    }

    /// Colour-convert, dither and encode `rows` (page lines from `first_line`).
    ///
    /// Same arguments as `_band_fast.render_band`: rows, skip (or empty to
    /// skip pure-white rows), width, first_line, sw, lut, grid, weights,
    /// black, thr_k, thr_c, thr_m, thr_y, lengths. The GIL is released while
    /// the band renders; the caller must not change the buffers meanwhile.
    ///
    /// Returns:
    ///     The encoded plane lines back to back, in line then plane order.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn render_band<'py>(
        py: Python<'py>,
        rows: PyBuffer<u8>,
        skip: PyBuffer<u8>,
        width: isize,
        first_line: isize,
        sw: isize,
        lut: PyBuffer<u8>,
        grid: PyBuffer<i32>,
        weights: PyBuffer<u8>,
        black: PyBuffer<i32>,
        thr_k: PyBuffer<u8>,
        thr_c: PyBuffer<u8>,
        thr_m: PyBuffer<u8>,
        thr_y: PyBuffer<u8>,
        mut lengths: PyBuffer<i32>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let (row_slices, row_len) = byte_rows(&rows)?;
        let skip = contiguous(&skip)?;
        let lut = contiguous(&lut)?;
        let (grid, weights, black) = (
            contiguous(&grid)?,
            contiguous(&weights)?,
            contiguous(&black)?,
        );
        let thresholds = [
            thresholds(&thr_k)?,
            thresholds(&thr_c)?,
            thresholds(&thr_m)?,
            thresholds(&thr_y)?,
        ];
        let (lengths, len_rows, len_cols) = matrix_mut(&mut lengths)?;

        let n = row_slices.len();
        let npx = width.min(sw).max(0) as usize;
        let skip = (!skip.is_empty()).then_some(skip);
        if skip.is_some_and(|s| s.len() != n) || len_rows != n || len_cols != 4 {
            return err("skip and lengths must have one entry per row");
        }
        if n > 0 && (row_len as isize) < width.saturating_mul(3) {
            return err("rows shorter than width * 3");
        }
        let has_lut = !lut.is_empty();
        if has_lut && lut.len() != LUT_SIZE {
            return err("lut must be the flat 256*256*256*4 inverse LUT");
        }
        if !has_lut
            && (grid.len() != GRID_ENTRIES * 4 || weights.len() != WEIGHT_BYTES || black.len() != 4)
        {
            return err("without an inverse LUT, grid, weights and black ink are required");
        }
        if thresholds.iter().any(|t| t.rows == 0 || t.cols < npx) {
            return err("threshold matrix narrower than the image");
        }
        let (width, first_line, sw) = geometry(width, first_line, sw)?;

        let colour = if has_lut {
            Colour::Lut(lut)
        } else {
            Colour::Grid(Grid {
                grid,
                weights,
                black,
            })
        };
        let band = Band {
            rows: &row_slices,
            skip,
            width,
            first_line,
            sw,
            colour,
            thresholds,
        };
        let out = py.detach(|| super::render_band(&band, lengths));
        Ok(PyBytes::new(py, &out))
    }

    /// Convert `rows` (page lines from `first_line`) to grayscale, dither and encode the K plane.
    ///
    /// Same arguments as `_band_fast.render_band_mono`: rows, width,
    /// first_line, sw, remap (768 bytes or empty), thr_k, lengths. The GIL
    /// is released while the band renders.
    ///
    /// Returns:
    ///     The encoded K lines back to back.
    #[pyfunction]
    #[allow(clippy::too_many_arguments)]
    fn render_band_mono<'py>(
        py: Python<'py>,
        rows: PyBuffer<u8>,
        width: isize,
        first_line: isize,
        sw: isize,
        remap: PyBuffer<u8>,
        thr_k: PyBuffer<u8>,
        mut lengths: PyBuffer<i32>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let (row_slices, row_len) = byte_rows(&rows)?;
        let remap = contiguous(&remap)?;
        let thresholds = thresholds(&thr_k)?;
        let lengths = contiguous_mut(&mut lengths)?;

        let n = row_slices.len();
        let npx = width.min(sw).max(0) as usize;
        if lengths.len() != n {
            return err("lengths must have one entry per row");
        }
        if n > 0 && row_len < npx * 3 {
            return err("rows shorter than width * 3");
        }
        let remap = match remap.len() {
            0 => None,
            768 => Some(remap.try_into().expect("768 bytes")),
            _ => return err("remap must be 3 x 256 bytes"),
        };
        if thresholds.rows == 0 || thresholds.cols < npx {
            return err("threshold matrix narrower than the image");
        }
        let (width, first_line, sw) = geometry(width, first_line, sw)?;

        let band = MonoBand {
            rows: &row_slices,
            width,
            first_line,
            sw,
            remap,
            thresholds,
        };
        let out = py.detach(|| super::render_band_mono(&band, lengths));
        Ok(PyBytes::new(py, &out))
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(render_band, m)?)?;
        m.add_function(wrap_pyfunction!(render_band_mono, m)?)
    }
}
