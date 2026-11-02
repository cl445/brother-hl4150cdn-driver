# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Render a band of RGB scanlines to four encoded 1bpp planes (Normal mode).

Colour lookup, ordered dither and sliding-window RLE for a whole band in
one call without the GIL, so bands can render on several threads. Output
is byte-identical to the per-line path in lines.encode_lines: colour
through the inverse LUT (or interpolated in the colour grid), 1bpp
threshold dither, then encode_plane_12bit / encode_plane_20bit / encode_plane_10bit.
`render_band_mono` does the same for grayscale pages, matching
lines.encode_lines_mono.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free, malloc
from libc.stdint cimport uint32_t, uint64_t
from libc.string cimport memcmp, memset

from brother_xl2hb._native._color_fast cimport interp_pixel
from brother_xl2hb._native._rle_fast cimport OutBuf, encode_line, encode_scratch_words

# RLE word size per plane (K, C, M, Y), as in
# plane_encoders.encode_plane_12bit / encode_plane_20bit / encode_plane_10bit.
cdef int[4] _BITS = [12, 20, 10, 12]

_LUT_SIZE = 256 * 256 * 256 * 4


cdef extern from *:
    """
    #include <string.h>
    #include <stdint.h>
    #if defined(__aarch64__)
    #include <arm_neon.h>
    #endif

    static inline uint64_t load_u64(const unsigned char *src) { uint64_t v; memcpy(&v, src, 8); return v; }
    static inline uint32_t load_u32(const unsigned char *src) { uint32_t v; memcpy(&v, src, 4); return v; }
    static inline void store_u32(unsigned char *dst, uint32_t v) { memcpy(dst, &v, 4); }

    /* R << 16 | G << 8 | B of the pixel at p. Reads 4 bytes, so not for a row's last pixel. */
    static inline uint32_t rgb_at(const unsigned char *p) {
        uint32_t v;
        memcpy(&v, p, 4);
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return __builtin_bswap32(v) >> 8;
    #else
        return v >> 8;
    #endif
    }

    /* 1bpp dither of the 16 interleaved KCMY pixels from pixel j (j a
       multiple of 8) against the threshold rows t[0..3]: bit set where
       255 - v > t, MSB first. Writes 2 bytes to each plane row out[0..3]
       at byte j / 8 and ORs them into ink[0..3]. */
    static inline void dither16_kcmy(const unsigned char *kcmy, const unsigned char *const t[4], Py_ssize_t j,
                                     unsigned char *const out[4], unsigned char ink[4]) {
    #if defined(__aarch64__)
        static const uint8_t weights[16] = {128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1};
        uint8x16_t w = vld1q_u8(weights);
        uint8x16x4_t v = vld4q_u8(kcmy + 4 * j);
        uint8x16_t k = vandq_u8(vcgtq_u8(vmvnq_u8(v.val[0]), vld1q_u8(t[0] + j)), w);
        uint8x16_t c = vandq_u8(vcgtq_u8(vmvnq_u8(v.val[1]), vld1q_u8(t[1] + j)), w);
        uint8x16_t m = vandq_u8(vcgtq_u8(vmvnq_u8(v.val[2]), vld1q_u8(t[2] + j)), w);
        uint8x16_t y = vandq_u8(vcgtq_u8(vmvnq_u8(v.val[3]), vld1q_u8(t[3] + j)), w);
        uint8x16_t q = vpaddq_u8(vpaddq_u8(k, c), vpaddq_u8(m, y));
        uint8x16_t r = vpaddq_u8(q, q);  /* bytes 0-1 K, 2-3 C, 4-5 M, 6-7 Y */
        uint8_t b[8];
        vst1_u8(b, vget_low_u8(r));
        for (int ch = 0; ch < 4; ch++) {
            out[ch][j >> 3] = b[2 * ch];
            out[ch][(j >> 3) + 1] = b[2 * ch + 1];
            ink[ch] |= b[2 * ch] | b[2 * ch + 1];
        }
    #else
        for (int ch = 0; ch < 4; ch++) {
            for (int h = 0; h < 2; h++) {
                unsigned char b = 0;
                for (int q = 0; q < 8; q++) {
                    Py_ssize_t x = j + 8 * h + q;
                    b |= (unsigned char)(((unsigned char)(255 - kcmy[4 * x + ch]) > t[ch][x]) << (7 - q));
                }
                out[ch][(j >> 3) + h] = b;
                ink[ch] |= b;
            }
        }
    #endif
    }

    /* 1bpp dither of 16 luma values: bit set where 255 - v > t, MSB first.
       Writes 2 bytes to out and returns their OR. */
    static inline unsigned char dither16_luma(const unsigned char *v, const unsigned char *t, unsigned char *out) {
    #if defined(__aarch64__)
        static const uint8_t weights[16] = {128, 64, 32, 16, 8, 4, 2, 1, 128, 64, 32, 16, 8, 4, 2, 1};
        uint8x16_t m = vandq_u8(vcgtq_u8(vmvnq_u8(vld1q_u8(v)), vld1q_u8(t)), vld1q_u8(weights));
        m = vpaddq_u8(m, m);
        m = vpaddq_u8(m, m);
        m = vpaddq_u8(m, m);
        out[0] = vgetq_lane_u8(m, 0);
        out[1] = vgetq_lane_u8(m, 1);
    #else
        for (int h = 0; h < 2; h++) {
            unsigned char b = 0;
            for (int q = 0; q < 8; q++)
                b |= (unsigned char)(((unsigned char)(255 - v[8 * h + q]) > t[8 * h + q]) << (7 - q));
            out[h] = b;
        }
    #endif
        return out[0] | out[1];
    }
    """
    uint64_t load_u64(const unsigned char *src) noexcept nogil
    uint32_t load_u32(const unsigned char *src) noexcept nogil
    void store_u32(unsigned char *dst, uint32_t v) noexcept nogil
    uint32_t rgb_at(const unsigned char *p) noexcept nogil
    void dither16_kcmy(
        const unsigned char *kcmy, const unsigned char **t, Py_ssize_t j, unsigned char **out, unsigned char *ink
    ) noexcept nogil
    unsigned char dither16_luma(const unsigned char *v, const unsigned char *t, unsigned char *out) noexcept nogil


cdef inline bint _row_white(const unsigned char *row, Py_ssize_t nbytes) noexcept nogil:
    cdef Py_ssize_t i = 0
    while i + 8 <= nbytes:
        if load_u64(row + i) != 0xFFFFFFFFFFFFFFFFULL:
            return False
        i += 8
    while i < nbytes:
        if row[i] != 255:
            return False
        i += 1
    return True


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

    Two passes per row: the KCMY values of every pixel (looked up once per
    run of equal pixels, and not at all for a row equal to the last
    converted one), then the threshold dither 16 pixels at a time. Plane
    lines without ink are not encoded; they would encode to nothing anyway.

    Args:
        rows: (n, >= width * 3) RGB scanlines; each row contiguous, any row
            stride (cropped or reversed views are fine).
        skip: n flags, a set flag emits a blank line without looking at it;
            or empty, to skip the rows that are pure white over `width`.
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
    cdef bint has_skip = skip.shape[0] != 0
    cdef bint has_ink = False
    cdef Py_ssize_t i, j, k, line, pid, written
    cdef uint32_t px, prev_px, value, no_ink
    cdef unsigned char kcmy4[4]
    cdef unsigned char ink[4]
    cdef const unsigned char *row
    cdef const unsigned char *prev_row = NULL
    cdef const unsigned char *t
    cdef const unsigned char *thr[4]
    cdef const unsigned char *trow[4]
    cdef Py_ssize_t thr_h[4]
    cdef Py_ssize_t thr_w[4]
    cdef unsigned char *planes
    cdef unsigned char *plane
    cdef unsigned char *prow[4]
    cdef unsigned char *kcmy
    cdef unsigned int *scratch
    cdef OutBuf o

    if (has_skip and skip.shape[0] != n) or lengths.shape[0] != n or lengths.shape[1] != 4:
        raise ValueError("skip and lengths must have one entry per row")
    if n and rows.shape[1] < width * 3:
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

    thr[0] = &thr_k[0, 0]
    thr[1] = &thr_c[0, 0]
    thr[2] = &thr_m[0, 0]
    thr[3] = &thr_y[0, 0]
    thr_h[0] = thr_k.shape[0]
    thr_h[1] = thr_c.shape[0]
    thr_h[2] = thr_m.shape[0]
    thr_h[3] = thr_y.shape[0]
    thr_w[0] = thr_k.shape[1]
    thr_w[1] = thr_c.shape[1]
    thr_w[2] = thr_m.shape[1]
    thr_w[3] = thr_y.shape[1]

    planes = <unsigned char *>malloc(4 * bpl)
    kcmy = <unsigned char *>malloc(4 * npx + 4)
    scratch = <unsigned int *>malloc(encode_scratch_words(bpl, 10) * sizeof(unsigned int))
    o.data = NULL
    o.len = 0
    o.cap = 0
    o.failed = planes == NULL or kcmy == NULL or scratch == NULL
    for pid in range(4):
        prow[pid] = planes + pid * bpl

    try:
        with nogil:
            for i in range(n):
                if o.failed:
                    break
                for pid in range(4):
                    lengths[i, pid] = 0
                row = &rows[i, 0]
                if has_skip:
                    if skip[i]:
                        continue
                elif _row_white(row, width * 3):
                    continue
                # Pass 1: KCMY per pixel, unless the row repeats the last converted one.
                if prev_row == NULL or memcmp(row, prev_row, npx * 3) != 0:
                    prev_px = 0xFFFFFFFFU
                    value = 0
                    # AND of all KCMY values; all ones means no channel has ink.
                    no_ink = 0xFFFFFFFFU
                    for j in range(npx):
                        if j + 1 < npx:
                            px = rgb_at(row + 3 * j)
                        else:
                            px = (<uint32_t>row[3 * j] << 16) | (<uint32_t>row[3 * j + 1] << 8) | row[3 * j + 2]
                        if px != prev_px:
                            prev_px = px
                            if has_lut:
                                value = load_u32(&lut[(<size_t>px) << 2])
                            else:
                                interp_pixel(
                                    &grid[0], &weights[0], &black[0], row[3 * j], row[3 * j + 1], row[3 * j + 2], kcmy4
                                )
                                value = load_u32(kcmy4)
                        store_u32(kcmy + 4 * j, value)
                        no_ink &= value
                    prev_row = row
                    has_ink = no_ink != 0xFFFFFFFFU
                if not has_ink:
                    continue
                # Pass 2: threshold dither, 8 pixels per byte.
                line = first_line + i
                memset(planes, 0, 4 * bpl)
                for pid in range(4):
                    trow[pid] = thr[pid] + (line % thr_h[pid]) * thr_w[pid]
                    ink[pid] = 0
                j = 0
                while j + 16 <= npx:
                    dither16_kcmy(kcmy, trow, j, prow, ink)
                    j += 16
                for pid in range(4):
                    t = trow[pid]
                    plane = prow[pid]
                    for k in range(j, npx):
                        if <unsigned char>(255 - kcmy[4 * k + pid]) > t[k]:
                            plane[k >> 3] |= <unsigned char>(0x80 >> (k & 7))
                            ink[pid] = 1
                    if not ink[pid]:
                        continue
                    written = encode_line(plane, bpl, _BITS[pid], scratch, &o)
                    if written < 0:
                        break
                    lengths[i, pid] = <int>written
        if o.failed:
            raise MemoryError()
        return PyBytes_FromStringAndSize(<char *>o.data, o.len)
    finally:
        free(planes)
        free(kcmy)
        free(scratch)
        free(o.data)


def render_band_mono(
    const unsigned char[:, :] rows,
    Py_ssize_t width,
    Py_ssize_t first_line,
    Py_ssize_t sw,
    const unsigned char[::1] remap,
    const unsigned char[:, ::1] thr_k,
    int[::1] lengths,
):
    """Convert `rows` (page lines from `first_line`) to grayscale, dither and encode the K plane.

    K ink is 255 - luma, luma = (299 R + 587 G + 114 B + 499) // 1000 after
    the optional input remap; pure white (255, 255, 255) is not remapped.
    C, M and Y stay empty.

    Args:
        rows: (n, >= width * 3) RGB scanlines; each row contiguous, any row
            stride (cropped or reversed views are fine).
        width: pixels per input row.
        first_line: page line index of rows[0]; selects the threshold rows.
        sw: image width in pixels; pixels past `width` count as white.
        remap: 768 bytes, the R, G and B input remap tables back to back, or
            empty for none.
        thr_k: K threshold matrix tiled to >= sw columns.
        lengths: n outputs, encoded K length per line; 0 means no ink.

    Returns:
        The encoded K lines back to back.
    """
    cdef Py_ssize_t n = rows.shape[0]
    cdef Py_ssize_t npx = width if width < sw else sw
    cdef Py_ssize_t bpl = (sw + 7) // 8
    cdef bint has_remap = remap.shape[0] != 0
    cdef bint has_ink = False
    cdef Py_ssize_t i, j, written
    cdef unsigned int r, g, b, px, prev_px
    cdef unsigned char lv, acc
    cdef const unsigned char *row
    cdef const unsigned char *prev_row = NULL
    cdef const unsigned char *t
    cdef unsigned char *plane
    cdef unsigned char *luma
    cdef unsigned int *scratch
    cdef OutBuf o

    if lengths.shape[0] != n:
        raise ValueError("lengths must have one entry per row")
    if n and rows.shape[1] < npx * 3:
        raise ValueError("rows shorter than width * 3")
    if rows.shape[1] > 1 and rows.strides[1] != 1:
        raise ValueError("rows must be contiguous scanlines")
    if has_remap and remap.shape[0] != 768:
        raise ValueError("remap must be 3 x 256 bytes")
    if thr_k.shape[0] == 0 or thr_k.shape[1] < npx:
        raise ValueError("threshold matrix narrower than the image")
    if sw <= 0:
        raise ValueError("sw must be positive")

    # 16 bytes of slack: the dither reads and writes whole 16-pixel groups.
    plane = <unsigned char *>malloc(bpl + 16)
    luma = <unsigned char *>malloc(npx + 16)
    scratch = <unsigned int *>malloc(encode_scratch_words(bpl, 12) * sizeof(unsigned int))
    o.data = NULL
    o.len = 0
    o.cap = 0
    o.failed = plane == NULL or luma == NULL or scratch == NULL

    try:
        with nogil:
            for i in range(n):
                if o.failed:
                    break
                lengths[i] = 0
                row = &rows[i, 0]
                if _row_white(row, npx * 3):
                    continue
                # Luma of the row, unless it repeats the last converted one.
                if prev_row == NULL or memcmp(row, prev_row, npx * 3) != 0:
                    prev_px = 0xFFFFFFFFU
                    lv = 255
                    has_ink = False
                    for j in range(npx):
                        r = row[3 * j]
                        g = row[3 * j + 1]
                        b = row[3 * j + 2]
                        px = (r << 16) | (g << 8) | b
                        if px != prev_px:
                            prev_px = px
                            if has_remap and px != 0xFFFFFF:
                                r = remap[r]
                                g = remap[256 + g]
                                b = remap[512 + b]
                            lv = <unsigned char>((299 * r + 587 * g + 114 * b + 499) // 1000)
                            if lv != 255:
                                has_ink = True
                        luma[j] = lv
                    prev_row = row
                if not has_ink:
                    continue
                t = &thr_k[(first_line + i) % thr_k.shape[0], 0]
                memset(plane, 0, bpl)
                acc = 0
                j = 0
                while j + 16 <= npx:
                    acc |= dither16_luma(luma + j, t + j, plane + (j >> 3))
                    j += 16
                while j < npx:
                    if <unsigned char>(255 - luma[j]) > t[j]:
                        plane[j >> 3] |= <unsigned char>(0x80 >> (j & 7))
                        acc = 1
                    j += 1
                if acc == 0:
                    continue
                written = encode_line(plane, bpl, 12, scratch, &o)
                if written < 0:
                    break
                lengths[i] = <int>written
        if o.failed:
            raise MemoryError()
        return PyBytes_FromStringAndSize(<char *>o.data, o.len)
    finally:
        free(plane)
        free(luma)
        free(scratch)
        free(o.data)
