//! Render a band of RGB scanlines to four encoded 1bpp planes (Normal mode).
//!
//! Colour lookup, ordered dither and sliding-window RLE for a whole band in
//! one call without the GIL, so bands can render on several threads. Output
//! is byte-identical to the per-line path in lines.encode_lines: colour
//! through the inverse LUT (or interpolated in the colour grid), 1bpp
//! threshold dither, then encode_plane_12bit / encode_plane_20bit /
//! encode_plane_10bit. Port of `_band_fast.pyx`.

use crate::color::Grid;
use crate::rle::encode_line;

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

/// One band: the inputs `render_band` checked.
pub struct Band<'a> {
    /// RGB scanlines, each at least `min(width, sw) * 3` bytes.
    pub rows: &'a [&'a [u8]],
    /// Per row: a set flag emits a blank line without looking at it.
    pub skip: &'a [u8],
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

/// Dither one scanline's four planes into `planes` (4 x bpl bytes, K, C, M, Y).
///
/// `ink` maps one RGB pixel to its K, C, M, Y; generic so that each colour
/// source gets its own pixel loop.
#[inline(always)]
fn dither_line(
    band: &Band<'_>,
    ink: impl Fn(&[u8]) -> [u8; 4],
    row: &[u8],
    line: usize,
    planes: &mut [u8],
    bpl: usize,
) {
    let npx = band.width.min(band.sw);
    planes.fill(0);
    let thr = band.thresholds.map(|t| t.row(line, npx));
    let (pk, rest) = planes.split_at_mut(bpl);
    let (pc, rest) = rest.split_at_mut(bpl);
    let (pm, py) = rest.split_at_mut(bpl);
    for (n, px8) in row[..npx * 3].chunks(24).enumerate() {
        let mut acc = [0u8; 4];
        for (k, px) in px8.as_chunks::<3>().0.iter().enumerate() {
            let j = n * 8 + k;
            let ink = ink(px);
            let bit = 0x80 >> k;
            for p in 0..4 {
                if 255 - ink[p] > thr[p][j] {
                    acc[p] |= bit;
                }
            }
        }
        pk[n] = acc[0];
        pc[n] = acc[1];
        pm[n] = acc[2];
        py[n] = acc[3];
    }
}

/// `render_band` for one colour source.
#[inline(always)]
fn render_rows(
    band: &Band<'_>,
    ink: impl Fn(&[u8]) -> [u8; 4] + Copy,
    lengths: &mut [i32],
) -> Vec<u8> {
    let bpl = band.sw.div_ceil(8);
    let mut planes = vec![0u8; 4 * bpl];
    let mut out = Vec::new();
    for (i, (row, lens)) in band
        .rows
        .iter()
        .zip(lengths.as_chunks_mut::<4>().0.iter_mut())
        .enumerate()
    {
        if band.skip[i] != 0 {
            lens.fill(0);
            continue;
        }
        dither_line(band, ink, row, band.first_line + i, &mut planes, bpl);
        for (pid, len) in lens.iter_mut().enumerate() {
            *len = encode_line(&planes[pid * bpl..][..bpl], BITS[pid], &mut out) as i32;
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
            let ink = |px: &[u8]| {
                let rgb =
                    (usize::from(px[0]) << 16) | (usize::from(px[1]) << 8) | usize::from(px[2]);
                entries[rgb & 0xFF_FFFF]
            };
            render_rows(band, ink, lengths)
        }
        Colour::Grid(grid) => render_rows(
            band,
            |px: &[u8]| grid.interp_pixel(px[0], px[1], px[2]),
            lengths,
        ),
    }
}

#[cfg(feature = "python")]
pub use bindings::register;

#[cfg(feature = "python")]
mod bindings {
    use pyo3::buffer::PyBuffer;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use super::{Band, Colour, LUT_SIZE, Thresholds};
    use crate::buffer::{byte_rows, contiguous, matrix, matrix_mut};
    use crate::color::{GRID_ENTRIES, Grid, WEIGHT_BYTES};

    fn err<T>(msg: &str) -> PyResult<T> {
        Err(PyValueError::new_err(msg.to_owned()))
    }

    /// Colour-convert, dither and encode `rows` (page lines from `first_line`).
    ///
    /// Same arguments as `_band_fast.render_band`: rows, skip, width,
    /// first_line, sw, lut, grid, weights, black, thr_k, thr_c, thr_m,
    /// thr_y, lengths. The GIL is released while the band renders; the
    /// caller must not change the buffers meanwhile.
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
        let thresholds = [&thr_k, &thr_c, &thr_m, &thr_y]
            .map(|t| matrix(t).map(|(data, rows, cols)| Thresholds { data, rows, cols }));
        let [tk, tc, tm, ty] = thresholds;
        let thresholds = [tk?, tc?, tm?, ty?];
        let (lengths, len_rows, len_cols) = matrix_mut(&mut lengths)?;

        let n = row_slices.len();
        let npx = width.min(sw).max(0) as usize;
        if skip.len() != n || len_rows != n || len_cols != 4 {
            return err("skip and lengths must have one entry per row");
        }
        if n > 0 && row_len < npx * 3 {
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
        if sw <= 0 {
            return err("sw must be positive");
        }
        if width < 0 || first_line < 0 {
            return err("width and first_line must not be negative");
        }

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
            width: width as usize,
            first_line: first_line as usize,
            sw: sw as usize,
            colour,
            thresholds,
        };
        let out = py.detach(|| super::render_band(&band, lengths));
        Ok(PyBytes::new(py, &out))
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(render_band, m)?)
    }
}
