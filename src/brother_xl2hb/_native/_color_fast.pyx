# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Per-scanline RGB→KCMY: gather through the inverse LUT, or interpolate the grid.

One pass over the RGB row instead of numpy's index build + fancy-index +
de-interleave, which dominate the colour stage on ARM. Also the saturation
step before it, one pixel at a time instead of numpy's int64/float64
temporaries.
"""

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from libc.math cimport trunc


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


cdef inline unsigned int x87_quotient_product(unsigned int a, unsigned int b, unsigned int c) noexcept nogil:
    """trunc((a / b) * c) with both steps rounded to a 64-bit significand, like the x87 FPU.

    Mirrors saturation._x87_quotient_product for a, c < 2**24 and 0 < b < 2**24.
    """
    cdef unsigned long long m = a // b
    cdef unsigned long long rem = a % b
    cdef unsigned long long hi, lo, low, high, kept, dropped, half
    cdef int s = 0
    cdef int k = 0
    if a == 0:
        return 0
    # a / b as m / 2**s with 2**63 <= m < 2**64, by long division.
    while m < (1ULL << 63):
        rem <<= 1
        m <<= 1
        if rem >= b:
            m |= 1
            rem -= b
        s += 1
    if 2 * rem > b or (2 * rem == b and (m & 1)):
        m += 1
        if m == 0:
            m = 1ULL << 63
            s -= 1
    # m * c as the 128-bit high:low, then rounded back to 64 significant bits.
    hi = (m >> 32) * c
    lo = (m & 0xFFFFFFFFULL) * c
    low = (hi << 32) + lo
    high = (hi >> 32) + (low < (hi << 32))
    if high == 0:
        return <unsigned int>(low >> s) if s < 64 else 0
    while (high >> k) != 0:
        k += 1
    kept = (high << (64 - k)) | (low >> k)
    dropped = low & ((1ULL << k) - 1)
    half = 1ULL << (k - 1)
    if dropped > half or (dropped == half and (kept & 1)):
        kept += 1
        if kept == 0:
            kept = 1ULL << 63
            k += 1
    if k >= s:
        return <unsigned int>(kept << (k - s))
    return <unsigned int>(kept >> (s - k)) if s - k < 64 else 0


cdef inline unsigned char desaturate_channel(int v, int center, int sat_mode) noexcept nogil:
    """One channel of negative saturation: towards `center`, but not past it."""
    cdef double adjusted = trunc(v + sat_mode * <double>(v - center) / 50.0)
    if v > center and adjusted < center:
        adjusted = center
    elif v < center and adjusted > center:
        adjusted = center
    return <unsigned char>adjusted


def adjust_saturation(const unsigned char[::1] rgb, Py_ssize_t pixels, int sat_mode):
    """Adjust the saturation of `pixels` RGB pixels; see `saturation.adjust_saturation_py`.

    Returns:
        The adjusted pixels as bytes.
    """
    cdef Py_ssize_t i
    cdef int ch[3]
    cdef int lo_val, hi_val, lo_idx, hi_idx, mid_idx, half_range, boost, old_range, new_range, num
    cdef unsigned int offset
    cdef unsigned char *out
    if pixels < 0 or rgb.shape[0] < pixels * 3:
        raise ValueError("rgb shorter than pixels * 3")
    result = PyBytes_FromStringAndSize(<const char *>&rgb[0] if pixels else NULL, pixels * 3)
    if sat_mode == 0:
        return result
    out = <unsigned char *>PyBytes_AS_STRING(result)
    with nogil:
        for i in range(pixels):
            ch[0] = out[3 * i]
            ch[1] = out[3 * i + 1]
            ch[2] = out[3 * i + 2]
            if ch[0] == ch[1] and ch[2] == ch[1]:
                continue
            if sat_mode > 0 and (
                ch[0] == 0 or ch[1] == 0 or ch[2] == 0 or ch[0] == 255 or ch[1] == 255 or ch[2] == 255
            ):
                continue
            # Ties go to the first channel, as in the original (and numpy's argmin/argmax).
            lo_idx = 0
            hi_idx = 0
            if ch[1] < ch[lo_idx]:
                lo_idx = 1
            if ch[2] < ch[lo_idx]:
                lo_idx = 2
            if ch[1] > ch[hi_idx]:
                hi_idx = 1
            if ch[2] > ch[hi_idx]:
                hi_idx = 2
            lo_val = ch[lo_idx]
            hi_val = ch[hi_idx]
            half_range = (hi_val - lo_val) >> 1
            if sat_mode < 0:
                out[3 * i] = desaturate_channel(ch[0], lo_val + half_range, sat_mode)
                out[3 * i + 1] = desaturate_channel(ch[1], lo_val + half_range, sat_mode)
                out[3 * i + 2] = desaturate_channel(ch[2], lo_val + half_range, sat_mode)
                continue
            mid_idx = 3 - lo_idx - hi_idx
            boost = half_range * sat_mode // 50
            if boost > 255 - hi_val:
                boost = 255 - hi_val
            if boost > lo_val:
                boost = lo_val
            old_range = hi_val - lo_val
            new_range = old_range + 2 * boost
            num = new_range * (ch[mid_idx] - lo_val)
            # Only exact quotients can come out one below floor division.
            if num != 0 and num % old_range == 0:
                offset = x87_quotient_product(new_range, old_range, ch[mid_idx] - lo_val)
            else:
                offset = num // old_range
            out[3 * i + hi_idx] = <unsigned char>(hi_val + boost)
            out[3 * i + lo_idx] = <unsigned char>(lo_val - boost)
            out[3 * i + mid_idx] = <unsigned char>(lo_val - boost + offset)
    return result
