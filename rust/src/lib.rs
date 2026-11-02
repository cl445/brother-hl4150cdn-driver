//! Rust kernels for `brother_xl2hb`, built as `brother_xl2hb._native._rust`.
//!
//! Each submodule stands in for the Cython extension of the same name and
//! exports the same functions; `brother_xl2hb._native.load` prefers it.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
mod buffer;
pub mod dither;

#[cfg(feature = "python")]
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let dither_fast = PyModule::new(m.py(), "_dither_fast")?;
    dither::register(&dither_fast)?;
    m.add_submodule(&dither_fast)?;
    Ok(())
}
