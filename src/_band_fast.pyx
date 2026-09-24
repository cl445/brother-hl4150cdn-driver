# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Render a band of RGB scanlines to four encoded 1bpp planes (Normal mode).

Colour lookup, ordered dither and sliding-window RLE for a whole band in
one call without the GIL, so bands can render on several threads. Output
is byte-identical to the per-line path in pipeline._render_page: colour
through the inverse LUT (or interpolated in the colour grid), 1bpp
threshold dither, then encode_plane / encode_c_plane / encode_m_plane_10.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free, malloc
from libc.string cimport memset

from _color_fast cimport interp_pixel
from _rle_fast cimport OutBuf, encode_line, encode_scratch_words

# RLE word size per plane (K, C, M, Y), as in
# plane_encoders.encode_plane / encode_c_plane / encode_m_plane_10.
cdef int[4] _BITS = [12, 20, 10, 12]

_LUT_SIZE = 256 * 256 * 256 * 4


def render_band(
    const unsigned char[:, :] rows,
    const unsigned char[::1] skip,
    Py_ssize_t width,
    Py_ssize_t first_line,
    Py_ssize_t sw,
    const unsigned char[::1] lut,
    const int[::1] grid,
    const unsigned char[::1] weights,
    const int[::1] black,
    const unsigned char[:, ::1] thr_k,
    const unsigned char[:, ::1] thr_c,
    const unsigned char[:, ::1] thr_m,
    const unsigned char[:, ::1] thr_y,
    int[:, ::1] lengths,
):
    """Colour-convert, dither and encode `rows` (page lines from `first_line`).

    Args:
        rows: (n, >= width * 3) RGB scanlines; each row contiguous, any row
            stride (cropped or reversed views are fine).
        skip: n flags; a set flag emits a blank line without looking at it.
        width: pixels per input row.
        first_line: page line index of rows[0]; selects the threshold rows.
        sw: image width in pixels; pixels past `width` count as white.
        lut: flat (256*256*256*4) KCMY inverse LUT, or empty to interpolate
            `grid` instead.
        grid, weights, black: colour grid, interpolation weights and black
            ink from `color_lut.interp_arrays`; used when `lut` is empty.
        thr_k, thr_c, thr_m, thr_y: threshold matrices tiled to >= sw columns.
        lengths: (n, 4) output, encoded length per line and plane (K, C, M, Y);
            0 means the plane line has no ink.

    Returns:
        The encoded plane lines back to back, in line then plane order.
    """
    cdef Py_ssize_t n = rows.shape[0]
    cdef Py_ssize_t npx = width if width < sw else sw
    cdef Py_ssize_t bpl = (sw + 7) // 8
    cdef bint has_lut = lut.shape[0] != 0
    cdef Py_ssize_t i, j, line, pid, written
    cdef size_t off
    cdef unsigned char kv, cv, mv, yv, bit
    cdef unsigned char kcmy[4]
    cdef const unsigned char *row
    cdef const unsigned char *tk
    cdef const unsigned char *tc
    cdef const unsigned char *tm
    cdef const unsigned char *ty
    cdef unsigned char *planes
    cdef unsigned char *pk
    cdef unsigned char *pc
    cdef unsigned char *pm
    cdef unsigned char *py
    cdef unsigned int *scratch
    cdef OutBuf o

    if skip.shape[0] != n or lengths.shape[0] != n or lengths.shape[1] != 4:
        raise ValueError("skip and lengths must have one entry per row")
    if n and rows.shape[1] < npx * 3:
        raise ValueError("rows shorter than width * 3")
    if rows.shape[1] > 1 and rows.strides[1] != 1:
        raise ValueError("rows must be contiguous scanlines")
    if has_lut and lut.shape[0] != _LUT_SIZE:
        raise ValueError("lut must be the flat 256*256*256*4 inverse LUT")
    if not has_lut and (grid.shape[0] != 4913 * 4 or weights.shape[0] != 17 * 289 * 9 or black.shape[0] != 4):
        raise ValueError("without an inverse LUT, grid, weights and black ink are required")
    if (
        min(thr_k.shape[0], thr_c.shape[0], thr_m.shape[0], thr_y.shape[0]) == 0
        or min(thr_k.shape[1], thr_c.shape[1], thr_m.shape[1], thr_y.shape[1]) < npx
    ):
        raise ValueError("threshold matrix narrower than the image")
    if sw <= 0:
        raise ValueError("sw must be positive")

    planes = <unsigned char *>malloc(4 * bpl)
    scratch = <unsigned int *>malloc(encode_scratch_words(bpl, 10) * sizeof(unsigned int))
    o.data = NULL
    o.len = 0
    o.cap = 0
    o.failed = planes == NULL or scratch == NULL
    pk = planes
    pc = planes + bpl
    pm = planes + 2 * bpl
    py = planes + 3 * bpl

    try:
        with nogil:
            for i in range(n):
                if o.failed:
                    break
                if skip[i]:
                    for pid in range(4):
                        lengths[i, pid] = 0
                    continue
                line = first_line + i
                row = &rows[i, 0]
                tk = &thr_k[line % thr_k.shape[0], 0]
                tc = &thr_c[line % thr_c.shape[0], 0]
                tm = &thr_m[line % thr_m.shape[0], 0]
                ty = &thr_y[line % thr_y.shape[0], 0]
                memset(planes, 0, 4 * bpl)
                for j in range(npx):
                    if has_lut:
                        off = ((<size_t>row[3 * j] << 16) | (<size_t>row[3 * j + 1] << 8) | row[3 * j + 2]) << 2
                        kv = lut[off]
                        cv = lut[off + 1]
                        mv = lut[off + 2]
                        yv = lut[off + 3]
                    else:
                        interp_pixel(&grid[0], &weights[0], &black[0], row[3 * j], row[3 * j + 1], row[3 * j + 2], kcmy)
                        kv = kcmy[0]
                        cv = kcmy[1]
                        mv = kcmy[2]
                        yv = kcmy[3]
                    bit = <unsigned char>(0x80 >> (j & 7))
                    if 255 - kv > tk[j]:
                        pk[j >> 3] |= bit
                    if 255 - cv > tc[j]:
                        pc[j >> 3] |= bit
                    if 255 - mv > tm[j]:
                        pm[j >> 3] |= bit
                    if 255 - yv > ty[j]:
                        py[j >> 3] |= bit
                for pid in range(4):
                    written = encode_line(planes + pid * bpl, bpl, _BITS[pid], scratch, &o)
                    if written < 0:
                        break
                    lengths[i, pid] = <int>written
        if o.failed:
            raise MemoryError()
        return PyBytes_FromStringAndSize(<char *>o.data, o.len)
    finally:
        free(planes)
        free(scratch)
        free(o.data)
