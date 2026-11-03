# Shared with _band_fast and _fine_fast: tetrahedral interpolation through one
# 17x17x17 colour grid, callable without the GIL. Mirrors
# color_lut._rgb_to_cmyk_interp_arr (and cmyk_interpolate in the original
# driver), in the compact form of color_lut.interp_arrays.

from libc.stdint cimport uint64_t

cdef extern from *:
    """
    /* Grid offset, in bytes of C, M, Y, K, of the 8 cube corners in
       interpolation-weight order: entries 0, 17, 1, 18, 289, 306, 290, 307. */
    static const int brxl2hb_corner_offsets[8] = {0, 68, 4, 72, 1156, 1224, 1160, 1228};
    """
    const int *brxl2hb_corner_offsets


cdef inline void interp_pixel(
    const unsigned char *grid,
    const unsigned char *weights,
    const uint64_t *recip,
    const int *black,
    unsigned char r,
    unsigned char g,
    unsigned char b,
    unsigned char *kcmy,
) noexcept nogil:
    """Write K, C, M, Y (0 = full ink, 255 = none) for one RGB pixel.

    `grid` holds 4913 entries of C, M, Y, K ink as bytes; `weights` 4913
    entries of 8 bytes: the 4 non-zero corner weights, their total, and the
    4 corner numbers (0-7) packed 3 bits each into bytes 5 and 6. `recip`
    holds ceil(2**32 / total) per total, so (n * recip) >> 32 is n // total;
    `black` the C, M, Y, K ink for pure black.
    """
    cdef int r_hi, g_hi, b_hi, r_frac, g_frac, b_frac, base, half, corners, wk, k
    cdef int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0
    cdef uint64_t m
    cdef const unsigned char *w
    cdef const unsigned char *corner
    if r == 255 and g == 255 and b == 255:
        kcmy[0] = 255
        kcmy[1] = 255
        kcmy[2] = 255
        kcmy[3] = 255
        return
    if r == 0 and g == 0 and b == 0:
        kcmy[0] = <unsigned char>(255 - black[3])
        kcmy[1] = <unsigned char>(255 - black[0])
        kcmy[2] = <unsigned char>(255 - black[1])
        kcmy[3] = <unsigned char>(255 - black[2])
        return
    if r == 255:
        r_hi = 15
        r_frac = 16
    else:
        r_hi = r >> 4
        r_frac = r & 15
    if g == 255:
        g_hi = 15
        g_frac = 16
    else:
        g_hi = g >> 4
        g_frac = g & 15
    if b == 255:
        b_hi = 15
        b_frac = 16
    else:
        b_hi = b >> 4
        b_frac = b & 15
    w = weights + (b_frac * 289 + g_frac * 17 + r_frac) * 8
    base = (r_hi * 289 + g_hi * 17 + b_hi) * 4
    corners = w[5] | (w[6] << 8)
    for k in range(4):
        wk = w[k]
        corner = grid + base + brxl2hb_corner_offsets[(corners >> (3 * k)) & 7]
        acc0 += wk * corner[0]
        acc1 += wk * corner[1]
        acc2 += wk * corner[2]
        acc3 += wk * corner[3]
    # A total of 0 divides by 1, as in the numpy path; its half is 0.
    half = w[4] >> 1
    m = recip[w[4]]
    # Ink → pixel-brightness; the uint8 casts wrap like the numpy path.
    kcmy[0] = <unsigned char>(255 - <int>(((acc3 + half) * m) >> 32))
    kcmy[1] = <unsigned char>(255 - <int>(((acc0 + half) * m) >> 32))
    kcmy[2] = <unsigned char>(255 - <int>(((acc1 + half) * m) >> 32))
    kcmy[3] = <unsigned char>(255 - <int>(((acc2 + half) * m) >> 32))
