# Shared with _band_fast: tetrahedral interpolation through one 17x17x17 colour
# grid, callable without the GIL. Mirrors color_lut._rgb_to_cmyk_interp_arr
# (and cmyk_interpolate in the original driver).

cdef inline void interp_pixel(
    const int *grid,
    const unsigned char *weights,
    const int *black,
    unsigned char r,
    unsigned char g,
    unsigned char b,
    unsigned char *kcmy,
) noexcept nogil:
    """Write K, C, M, Y (0 = full ink, 255 = none) for one RGB pixel.

    `grid` holds 4913 entries of C, M, Y, K ink; `weights` the 17 tables of
    289 x 9 bytes; `black` the C, M, Y, K ink for pure black.
    """
    cdef int r_hi, g_hi, b_hi, r_frac, g_frac, b_frac, base, total, half, ch, k
    cdef int acc[4]
    # Grid entry offsets of the 8 cube corners, in interpolation-weight order.
    cdef int corners[8]
    cdef const unsigned char *w
    cdef const int *corner
    corners[:] = [0, 17, 1, 18, 289, 306, 290, 307]
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
    w = weights + (b_frac * 289 + g_frac * 17 + r_frac) * 9
    total = w[0]
    if total == 0:
        total = 1
    base = r_hi * 289 + g_hi * 17 + b_hi
    for ch in range(4):
        acc[ch] = 0
    for k in range(8):
        corner = grid + (base + corners[k]) * 4
        for ch in range(4):
            acc[ch] += w[k + 1] * corner[ch]
    half = total >> 1
    # Ink → pixel-brightness; the uint8 casts wrap like the numpy path.
    kcmy[0] = <unsigned char>(255 - (acc[3] + half) // total)
    kcmy[1] = <unsigned char>(255 - (acc[0] + half) // total)
    kcmy[2] = <unsigned char>(255 - (acc[1] + half) // total)
    kcmy[3] = <unsigned char>(255 - (acc[2] + half) // total)
