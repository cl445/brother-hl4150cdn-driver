//! Views of Python buffers: strided and contiguous rows, matrices, a writable output.

use pyo3::buffer::{Element, PyBuffer};
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
        (self.stride == 1).then(|| bytes(self.buf, self.len))
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

/// A C-contiguous one-dimensional buffer as a slice, like Cython's `const T[::1]`.
pub fn contiguous<T: Element>(buf: &PyBuffer<T>) -> PyResult<&[T]> {
    if buf.dimensions() != 1 || !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "expected a contiguous one-dimensional buffer",
        ));
    }
    items(buf, buf.item_count())
}

/// A C-contiguous two-dimensional buffer as (items, rows, columns), like Cython's `const T[:, ::1]`.
pub fn matrix<T: Element>(buf: &PyBuffer<T>) -> PyResult<(&[T], usize, usize)> {
    if buf.dimensions() != 2 || !buf.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "expected a contiguous two-dimensional buffer",
        ));
    }
    Ok((
        items(buf, buf.item_count())?,
        buf.shape()[0],
        buf.shape()[1],
    ))
}

/// A writable C-contiguous two-dimensional buffer as (items, rows, columns), like Cython's `T[:, ::1]`.
pub fn matrix_mut<T: Element>(buf: &mut PyBuffer<T>) -> PyResult<(&mut [T], usize, usize)> {
    if buf.readonly() {
        return Err(PyValueError::new_err("expected a writable buffer"));
    }
    let (items, rows, cols) = matrix(buf)?;
    // SAFETY: the exporter allows writes, and the exclusive borrow of `buf`
    // makes this the only view of the buffer.
    let items = unsafe { std::slice::from_raw_parts_mut(items.as_ptr() as *mut T, items.len()) };
    Ok((items, rows, cols))
}

/// The rows of a two-dimensional byte buffer whose rows are contiguous, like
/// Cython's `const unsigned char[:, :]` with unit column stride; the row
/// stride may be anything (cropped and reversed views).
pub fn byte_rows(buf: &PyBuffer<u8>) -> PyResult<(Vec<&[u8]>, usize)> {
    if buf.dimensions() != 2 {
        return Err(PyValueError::new_err("expected a two-dimensional buffer"));
    }
    let (n, len) = (buf.shape()[0], buf.shape()[1]);
    if len > 1 && buf.strides()[1] != 1 {
        return Err(PyValueError::new_err("rows must be contiguous scanlines"));
    }
    let base = buf.buf_ptr() as *const u8;
    let stride = buf.strides()[0];
    let rows = (0..n)
        .map(|i| {
            if len == 0 {
                return &[][..];
            }
            // SAFETY: row i starts `i * stride` bytes from the first and holds
            // `len` contiguous bytes; the buffer stays exported while `buf` lives.
            unsafe { std::slice::from_raw_parts(base.offset(i as isize * stride), len) }
        })
        .collect();
    Ok((rows, len))
}

/// The first `len` items of a buffer that holds at least `len` contiguous items.
fn items<T: Element>(buf: &PyBuffer<T>, len: usize) -> PyResult<&[T]> {
    if len == 0 {
        // An empty exporter may report a null pointer, which a slice must not have.
        return Ok(&[]);
    }
    let ptr = buf.buf_ptr() as *const T;
    if !ptr.is_aligned() {
        return Err(PyValueError::new_err(
            "buffer is not aligned for its item type",
        ));
    }
    // SAFETY: the pointer is non-null and aligned, the buffer holds `len`
    // contiguous items and stays exported (so unchanged in size) while
    // `buf` lives.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// The first `len` bytes of a byte buffer that holds at least `len` contiguous bytes.
fn bytes(buf: &PyBuffer<u8>, len: usize) -> &[u8] {
    items(buf, len).expect("bytes are always aligned")
}
