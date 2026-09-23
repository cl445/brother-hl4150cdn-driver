# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Ordered 1bpp dither of one plane scanline against a tiled threshold row."""

from cpython.bytes cimport PyBytes_FromStringAndSize


def dither_row_1bpp(const unsigned char[:] row, const unsigned char[::1] thresholds, Py_ssize_t width):
    """Set a dot where ink (255 - pixel) exceeds the threshold; pack MSB-first.

    Returns:
        ceil(width / 8) bytes.
    """
    cdef Py_ssize_t bpl = (width + 7) // 8
    cdef Py_ssize_t i
    cdef unsigned char acc
    if row.shape[0] < width or thresholds.shape[0] < width:
        raise ValueError("row or thresholds shorter than width")
    cdef bytes out = PyBytes_FromStringAndSize(NULL, bpl)
    cdef unsigned char *po = <unsigned char *><char *>out
    for i in range(bpl):
        po[i] = 0
    with nogil:
        for i in range(width):
            if 255 - row[i] > thresholds[i]:
                po[i >> 3] |= <unsigned char>(0x80 >> (i & 7))
    return out
