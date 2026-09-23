# Shared with _band_fast: the sliding-window RLE core, callable without the GIL.

from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memcpy


cdef struct OutBuf:
    unsigned char *data
    Py_ssize_t len
    Py_ssize_t cap
    bint failed


cdef Py_ssize_t encode_line(
    const unsigned char *p,
    Py_ssize_t n_bytes,
    int read_group,
    int encode_group,
    int bits,
    unsigned int *scratch,
    OutBuf *o,
) noexcept nogil

cdef Py_ssize_t encode_scratch_words(Py_ssize_t n_bytes, int encode_group) noexcept nogil
