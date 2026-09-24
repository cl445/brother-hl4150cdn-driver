# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Per-scanline RGB→KCMY: gather through the inverse LUT, or interpolate the grid.

One pass over the RGB row instead of numpy's index build + fancy-index +
de-interleave, which dominate the colour stage on ARM.
"""


def gather_kcmy(
    const unsigned char[::1] rgb,
    Py_ssize_t width,
    const unsigned char[::1] lut,
    unsigned char[::1] k,
    unsigned char[::1] c,
    unsigned char[::1] m,
    unsigned char[::1] y,
):
    """Look up `width` RGB pixels in the flat (256*256*256*4) KCMY LUT.

    Writes the four channels into the preallocated `k`, `c`, `m`, `y`.
    """
    cdef Py_ssize_t i
    cdef size_t off
    if rgb.shape[0] < width * 3:
        raise ValueError("rgb row shorter than width * 3")
    if lut.shape[0] != 256 * 256 * 256 * 4:
        raise ValueError("lut must be the flat 256*256*256*4 inverse LUT")
    if k.shape[0] < width or c.shape[0] < width or m.shape[0] < width or y.shape[0] < width:
        raise ValueError("output planes shorter than width")
    with nogil:
        for i in range(width):
            off = ((<size_t>rgb[3 * i] << 16) | (<size_t>rgb[3 * i + 1] << 8) | rgb[3 * i + 2]) << 2
            k[i] = lut[off]
            c[i] = lut[off + 1]
            m[i] = lut[off + 2]
            y[i] = lut[off + 3]


def interp_kcmy(
    const unsigned char[::1] rgb,
    Py_ssize_t width,
    const int[::1] grid,
    const unsigned char[::1] weights,
    const int[::1] black,
    unsigned char[::1] k,
    unsigned char[::1] c,
    unsigned char[::1] m,
    unsigned char[::1] y,
):
    """Interpolate `width` RGB pixels through a 17x17x17 colour grid.

    `grid`, `weights` and `black` come from `color_lut.interp_arrays`.
    Writes the four channels into the preallocated `k`, `c`, `m`, `y`.
    """
    cdef Py_ssize_t i
    cdef unsigned char kcmy[4]
    if rgb.shape[0] < width * 3:
        raise ValueError("rgb row shorter than width * 3")
    if grid.shape[0] != 4913 * 4 or weights.shape[0] != 17 * 289 * 9 or black.shape[0] != 4:
        raise ValueError("grid, weights or black ink has the wrong size")
    if k.shape[0] < width or c.shape[0] < width or m.shape[0] < width or y.shape[0] < width:
        raise ValueError("output planes shorter than width")
    with nogil:
        for i in range(width):
            interp_pixel(&grid[0], &weights[0], &black[0], rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], kcmy)
            k[i] = kcmy[0]
            c[i] = kcmy[1]
            m[i] = kcmy[2]
            y[i] = kcmy[3]
