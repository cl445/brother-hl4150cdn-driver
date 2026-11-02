//! Read-only views of one-dimensional Python byte buffers.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A one-dimensional byte buffer that may be strided, like Cython's `const unsigned char[:]`.
pub struct ByteView<'a> {
    buf: &'a PyBuffer<u8>,
    len: usize,
    stride: isize,
}

impl<'a> ByteView<'a> {
    pub fn new(buf: &'a PyBuffer<u8>) -> PyResult<Self> {
        if buf.dimensions() != 1 {
            return Err(PyValueError::new_err("expected a one-dimensional buffer"));
        }
        Ok(Self {
            buf,
            len: buf.shape()[0],
            stride: buf.strides()[0],
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// The bytes as a slice if they are contiguous.
    pub fn as_slice(&self) -> Option<&[u8]> {
        // SAFETY: the buffer holds `len` contiguous bytes and stays exported
        // (so unchanged in size) while `buf` lives.
        (self.stride == 1).then(|| unsafe {
            std::slice::from_raw_parts(self.buf.buf_ptr() as *const u8, self.len)
        })
    }

    /// Byte `i`, which the caller keeps below `len`.
    ///
    /// # Safety
    /// `i` must be less than `len`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, i: usize) -> u8 {
        // SAFETY: i < len, and the buffer holds `len` items `stride` bytes apart.
        unsafe { *(self.buf.buf_ptr() as *const u8).offset(i as isize * self.stride) }
    }
}

/// A C-contiguous byte buffer as a slice, like Cython's `const unsigned char[::1]`.
pub fn contiguous(buf: &PyBuffer<u8>) -> PyResult<&[u8]> {
    if buf.dimensions() != 1 || !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "expected a contiguous one-dimensional buffer",
        ));
    }
    // SAFETY: the buffer is contiguous, holds item_count bytes and stays
    // exported (so unchanged in size) while `buf` lives.
    Ok(unsafe { std::slice::from_raw_parts(buf.buf_ptr() as *const u8, buf.item_count()) })
}
