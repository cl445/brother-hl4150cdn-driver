//! Ordered 1bpp dither of one plane scanline against a tiled threshold row.

/// Set a dot where ink (255 - pixel) exceeds the threshold; pack MSB-first into `out`.
///
/// `pixel(i)` reads pixel i for i < thresholds.len(); `out` holds
/// thresholds.len().div_ceil(8) bytes.
#[inline(always)]
pub fn dither_1bpp(pixel: impl Fn(usize) -> u8, thresholds: &[u8], out: &mut [u8]) {
    assert_eq!(out.len(), thresholds.len().div_ceil(8));
    for (n, (byte, chunk)) in out.iter_mut().zip(thresholds.chunks(8)).enumerate() {
        let mut acc = 0u8;
        for (k, &threshold) in chunk.iter().enumerate() {
            acc |= u8::from(255 - pixel(n * 8 + k) > threshold) << (7 - k);
        }
        *byte = acc;
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

    use super::dither_1bpp;
    use crate::buffer::{ByteView, contiguous};

    /// Set a dot where ink (255 - pixel) exceeds the threshold; pack MSB-first.
    ///
    /// Returns:
    ///     ceil(width / 8) bytes.
    #[pyfunction]
    fn dither_row_1bpp<'py>(
        py: Python<'py>,
        row: PyBuffer<u8>,
        thresholds: PyBuffer<u8>,
        width: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let row = ByteView::new(&row)?;
        let thresholds = contiguous(&thresholds)?;
        if row.len() < width || thresholds.len() < width {
            return Err(PyValueError::new_err(
                "row or thresholds shorter than width",
            ));
        }
        PyBytes::new_with(py, width.div_ceil(8), |out| {
            let thresholds = &thresholds[..width];
            // SAFETY: dither_1bpp reads pixels i < width <= row.len() only.
            py.detach(|| match row.as_slice() {
                Some(row) => dither_1bpp(|i| unsafe { *row.get_unchecked(i) }, thresholds, out),
                None => dither_1bpp(|i| unsafe { row.get_unchecked(i) }, thresholds, out),
            });
            Ok(())
        })
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(dither_row_1bpp, m)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_msb_first_and_zero_pads() {
        // Ink 255 beats every threshold but 255; ink 0 beats none.
        let row = [0u8, 255, 0, 0, 0, 0, 0, 0, 0, 0];
        let thresholds = [0u8, 0, 254, 255, 0, 0, 0, 0, 0, 0];
        let mut out = [0xffu8; 2];
        dither_1bpp(|i| row[i], &thresholds, &mut out);
        assert_eq!(out, [0b1010_1111, 0b1100_0000]);
    }
}
