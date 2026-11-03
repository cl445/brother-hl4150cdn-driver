//! Tetrahedral interpolation through one 17x17x17 colour grid.
//!
//! Mirrors color_lut._rgb_to_cmyk_interp_arr (and cmyk_interpolate in the
//! original driver), like `interp_pixel` in `_color_fast.pxd`.

/// Grid entries.
pub const GRID_ENTRIES: usize = 17 * 17 * 17;
/// Bytes of the interpolation weight tables: 17 tables of 289 x 9.
pub const WEIGHT_BYTES: usize = 17 * 289 * 9;

/// Grid entry offsets of the 8 cube corners, in interpolation-weight order.
const CORNERS: [usize; 8] = [0, 17, 1, 18, 289, 306, 290, 307];

/// The colour grid and its interpolation tables, from `color_lut.interp_arrays`.
#[derive(Clone, Copy)]
pub struct Grid<'a> {
    /// GRID_ENTRIES entries of C, M, Y, K ink.
    pub grid: &'a [i32],
    /// WEIGHT_BYTES bytes: per fraction triple, the weight total then 8 corner weights.
    pub weights: &'a [u8],
    /// C, M, Y, K ink for pure black.
    pub black: &'a [i32],
}

/// Grid cell and fraction of one channel; 255 sits at the top of cell 15.
#[inline(always)]
fn split(v: u8) -> (usize, usize) {
    if v == 255 {
        (15, 16)
    } else {
        (usize::from(v >> 4), usize::from(v & 15))
    }
}

impl Grid<'_> {
    /// K, C, M, Y (0 = full ink, 255 = none) for one RGB pixel.
    #[inline]
    pub fn interp_pixel(&self, r: u8, g: u8, b: u8) -> [u8; 4] {
        if r == 255 && g == 255 && b == 255 {
            return [255; 4];
        }
        if r == 0 && g == 0 && b == 0 {
            let ink = |ch: usize| (255 - self.black[ch]) as u8;
            return [ink(3), ink(0), ink(1), ink(2)];
        }
        let (r_hi, r_frac) = split(r);
        let (g_hi, g_frac) = split(g);
        let (b_hi, b_frac) = split(b);
        let w = &self.weights[(b_frac * 289 + g_frac * 17 + r_frac) * 9..][..9];
        let total = i32::from(w[0]).max(1);
        let base = r_hi * 289 + g_hi * 17 + b_hi;
        let mut acc = [0i32; 4];
        for (k, &corner) in CORNERS.iter().enumerate() {
            let entry = &self.grid[(base + corner) * 4..][..4];
            for ch in 0..4 {
                acc[ch] += i32::from(w[k + 1]) * entry[ch];
            }
        }
        let half = total >> 1;
        // Ink to pixel brightness; the u8 casts wrap like the numpy path.
        let level = |ch: usize| (255 - (acc[ch] + half) / total) as u8;
        [level(3), level(0), level(1), level(2)]
    }
}
