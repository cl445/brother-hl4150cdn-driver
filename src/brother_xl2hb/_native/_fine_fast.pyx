# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Native Fine mode (1200 dpi) codec and 4bpp dither.

Byte-identical counterparts of fine_encoder.compress_rle_preencode
(stage 1), fine_encoder.compress_pattern_encode (stage 2) and
fine_encoder.encode_fine_plane, plus the nibble-table lookup of
dither.dither_channel_4bpp_arr. The Python versions stay the reference;
the cores run without the GIL.

`render_band_fine` is the Fine counterpart of _band_fast.render_band:
colour lookup, 4bpp dither, the overrun quirk and both codec stages for a
whole band, so bands can render on several threads.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memset

from brother_xl2hb._native._color_fast cimport interp_pixel

ctypedef unsigned char uchar


cdef inline Py_ssize_t _flush_run(uchar *dst, Py_ssize_t o, Py_ssize_t run, uchar base) noexcept nogil:
    # Runs longer than 8 go out as full 8-runs (base | 7) first.
    while run > 8:
        dst[o] = base | 0x07
        o += 1
        run -= 8
    dst[o] = <uchar>((run - 1) | base)
    return o + 1


cdef Py_ssize_t rle_preencode_c(const uchar *src, Py_ssize_t n, uchar *dst) noexcept nogil:
    """Stage 1 into `dst`, which must hold 2 * n + 16 bytes.

    Returns:
        Number of bytes written.
    """
    cdef Py_ssize_t i, o = 0
    cdef Py_ssize_t run_white = 0, run_black = 0
    cdef uchar v, hi, lo
    for i in range(n):
        v = src[i]
        if (v & 0xF0) != 0xF0:
            v &= 0xEF
        if (v & 0x0F) != 0x0F:
            v &= 0xFE

        if v == 0x00:
            if run_black == 0:
                run_white += 1
            else:
                if run_black < 5:
                    dst[o] = <uchar>(((run_black * 8 - 8) & 0xFF) | 0x80)
                    o += 1
                else:
                    o = _flush_run(dst, o, run_black, 0x18)
                    run_white += 1
                run_black = 0
        elif v == 0xFF:
            if run_white == 0:
                run_black += 1
            else:
                if run_white < 3:
                    dst[o] = <uchar>(((run_white * 8 - 8) & 0xFF) | 0x27)
                    o += 1
                else:
                    o = _flush_run(dst, o, run_white, 0x10)
                    run_black += 1
                run_white = 0
        else:
            hi = v & 0xF0
            lo = v & 0x0F
            if hi == 0xF0:
                if run_white == 0:
                    if run_black != 0:
                        if run_black > 3:
                            o = _flush_run(dst, o, run_black - 3, 0x18)
                            run_black = 3
                        dst[o] = <uchar>((run_black * 8) | (lo >> 1) | 0xC0)
                        o += 1
                        run_black = 0
                        continue
                else:
                    if lo == 0:
                        if run_white > 3:
                            o = _flush_run(dst, o, run_white - 3, 0x10)
                            run_white = 3
                        dst[o] = <uchar>((run_white * 8) | 0xA7)
                        o += 1
                        run_white = 0
                        continue
                    o = _flush_run(dst, o, run_white, 0x10)
                    run_white = 0
                dst[o] = <uchar>((lo >> 1) | 0xC0)
                o += 1
            elif lo == 0:
                if run_white != 0:
                    if run_white > 3:
                        o = _flush_run(dst, o, run_white - 3, 0x10)
                        run_white = 3
                    dst[o] = <uchar>((run_white * 8) | ((hi >> 5) - 1) | 0xA0)
                    o += 1
                    run_white = 0
                    continue
                if run_black != 0:
                    if run_black > 4:
                        o = _flush_run(dst, o, run_black - 4, 0x18)
                        run_black = 4
                    dst[o] = <uchar>(((run_black * 8 - 8) & 0xFF) | (hi >> 5) | 0x80)
                    o += 1
                    run_black = 0
                    continue
                dst[o] = <uchar>(((hi >> 5) - 1) | 0xA0)
                o += 1
            elif lo == 0x0F:
                if run_white != 0:
                    if run_white > 2:
                        o = _flush_run(dst, o, run_white - 2, 0x10)
                        run_white = 2
                    if hi == 0:
                        dst[o] = <uchar>(((run_white * 8 - 8) & 0xFF) | 0x37)
                    else:
                        dst[o] = <uchar>(((run_white * 8 - 8) & 0xFF) | ((hi >> 5) - 1) | 0x20)
                    o += 1
                    run_white = 0
                    continue
                if run_black != 0:
                    if run_black > 4:
                        o = _flush_run(dst, o, run_black - 4, 0x18)
                        run_black = 4
                    dst[o] = <uchar>(((run_black * 8 - 8) & 0xFF) | (hi >> 5) | 0xE0)
                    o += 1
                    run_black = 0
                    continue
                dst[o] = <uchar>((hi >> 2) | 0x47)
                o += 1
            elif hi == 0:
                if run_black == 0:
                    if run_white != 0:
                        if run_white > 2:
                            o = _flush_run(dst, o, run_white - 2, 0x10)
                            run_white = 2
                        dst[o] = <uchar>(((run_white * 8 - 8) & 0xFF) | ((lo >> 1) - 1) | 0x30)
                        o += 1
                        run_white = 0
                        continue
                else:
                    o = _flush_run(dst, o, run_black, 0x18)
                    run_black = 0
                dst[o] = <uchar>(((lo >> 1) - 1) | 0x40)
                o += 1
            else:
                if run_white == 0:
                    if run_black != 0:
                        o = _flush_run(dst, o, run_black, 0x18)
                        run_black = 0
                else:
                    o = _flush_run(dst, o, run_white, 0x10)
                    run_white = 0
                dst[o] = <uchar>((hi >> 2) | ((lo >> 1) - 1) | 0x40)
                o += 1

    if run_white != 0:
        o = _flush_run(dst, o, run_white, 0x10)
    elif run_black != 0:
        o = _flush_run(dst, o, run_black, 0x18)
    return o


cdef inline Py_ssize_t _emit_ext_count(uchar *dst, Py_ssize_t o, Py_ssize_t remaining) noexcept nogil:
    while remaining > 0xFE:
        dst[o] = 0xFF
        o += 1
        remaining -= 0xFF
    dst[o] = <uchar>(remaining & 0xFF)
    return o + 1


cdef Py_ssize_t _pattern_flush(
    uchar *dst,
    Py_ssize_t *po,
    int mode,
    Py_ssize_t count,
    uchar p0,
    uchar p1,
    uchar p2,
    const uchar *ring,
    Py_ssize_t start,
) noexcept nogil:
    """Write the pattern header and payload at dst[*po], advancing *po.

    The short headers are unreachable, as in fine_encoder._pattern_flush.

    Returns:
        Updated pattern start after the matched bytes.
    """
    cdef Py_ssize_t o = po[0]
    cdef Py_ssize_t k
    if mode == 0:
        if count - 3 < 3:
            dst[o] = <uchar>(count - 3)
            o += 1
        else:
            dst[o] = 3
            o = _emit_ext_count(dst, o + 1, count - 6)
        dst[o] = p0
        o += 1
        start += count
    elif mode == 1:
        if count == 3:
            dst[o] = 0x08
            o += 1
        else:
            dst[o] = 0x09
            o = _emit_ext_count(dst, o + 1, count - 4)
        dst[o] = p0
        o += 1
        for k in range(count):
            dst[o] = ring[start + 1]
            o += 1
            start += 2
    elif mode == 2:
        if count - 2 < 3:
            dst[o] = <uchar>((count - 2) | 0x04)
            o += 1
        else:
            dst[o] = 0x07
            o = _emit_ext_count(dst, o + 1, count - 5)
        dst[o] = p0
        dst[o + 1] = p1
        o += 2
        start += count * 2
    elif mode == 3:
        if count == 3:
            dst[o] = 0x0A
            o += 1
        else:
            dst[o] = 0x0B
            o = _emit_ext_count(dst, o + 1, count - 4)
        dst[o] = p0
        o += 1
        for k in range(count):
            dst[o] = ring[start + 1]
            dst[o + 1] = ring[start + 2]
            o += 2
            start += 3
    elif mode == 4:
        if count == 2:
            dst[o] = 0x0C
            o += 1
        else:
            dst[o] = 0x0D
            o = _emit_ext_count(dst, o + 1, count - 3)
        dst[o] = p0
        dst[o + 1] = p1
        o += 2
        for k in range(count):
            dst[o] = ring[start + 2]
            o += 1
            start += 3
    elif mode == 5:
        if count == 2:
            dst[o] = 0x0E
            o += 1
        else:
            dst[o] = 0x0F
            o = _emit_ext_count(dst, o + 1, count - 3)
        dst[o] = p0
        dst[o + 1] = p1
        dst[o + 2] = p2
        o += 3
        start += count * 3
    po[0] = o
    return start


cdef Py_ssize_t pattern_encode_c(const uchar *ring, Py_ssize_t n, uchar *dst) noexcept nogil:
    """Stage 2 into `dst`, which must hold 2 * n + 16 bytes.

    As in the Python reference, `ring` is the input itself rather than the
    binary's 1 KiB ring buffer.

    Returns:
        Number of bytes written.
    """
    cdef Py_ssize_t o = 0
    cdef Py_ssize_t i, buf_len, pp, s, k
    cdef Py_ssize_t start = 0
    cdef bint has_pattern = False, do_flush, skip_phase_reset, cont, check, literal
    cdef int mode = 0
    cdef Py_ssize_t count = 0, lit_count
    cdef int phase = 0
    cdef uchar v, p0 = 0, p1 = 0, p2 = 0
    cdef Py_ssize_t s1_len, s2_half, s3_a_len, s2_b_len, s3_b_len, s3_c_len, s3_alt_len

    for i in range(n):
        v = ring[i]
        buf_len = i + 1 - start
        do_flush = False
        skip_phase_reset = False

        if has_pattern:
            phase += 1
            cont = False
            check = False
            if mode == 0:
                if v == p0:
                    count += 1
                    phase = 0
                    check = True
            elif mode == 1:
                if phase == 1:
                    check = v == p0
                else:
                    count += 1
                    phase = 0
                    check = True
            elif mode == 2:
                if phase == 1:
                    check = v == p0
                elif v == p1:
                    count += 1
                    phase = 0
                    check = True
            elif mode == 3:
                if phase == 1:
                    check = v == p0
                else:
                    if phase != 2:
                        count += 1
                        phase = 0
                    check = True
            elif mode == 4:
                if phase == 1:
                    check = v == p0
                elif phase == 2:
                    check = v == p1
                else:
                    count += 1
                    phase = 0
                    check = True
            elif mode == 5:
                if phase == 1:
                    check = v == p0
                elif phase == 2:
                    check = v == p1
                elif v == p2:
                    count += 1
                    phase = 0
                    check = True
            cont = check and buf_len < 0x400
            if cont:
                skip_phase_reset = True
            else:
                do_flush = True
                has_pattern = False
        elif buf_len != 12:
            skip_phase_reset = True
        else:
            pp = start
            p0 = ring[pp]

            s1_len = 1
            while s1_len < 12 and ring[pp + s1_len] == p0:
                s1_len += 1

            s2_half = 2
            while s2_half < 12 and ring[pp + s2_half] == p0:
                s2_half += 2
            s2_half >>= 1

            s = 3
            s3_a_len = 1
            while s < 12 and ring[pp + s] == p0:
                s += 3
                s3_a_len += 1

            p1 = ring[pp + 1]

            s2_b_len = 1
            if s2_half > 1:
                s = 3
                while s < 12 and ring[pp + s] == p1:
                    s += 2
                s2_b_len = (s - 1) >> 1
                if s2_half < s2_b_len:
                    s2_b_len = s2_half

            p2 = ring[pp + 2]

            if s3_a_len < 2:
                s3_b_len = 1
                s3_c_len = 1
            else:
                s = 4
                s3_b_len = 1
                while s < 12 and ring[pp + s] == p1:
                    s += 3
                    s3_b_len += 1
                if s3_a_len < s3_b_len:
                    s3_b_len = s3_a_len
                s = 5
                s3_c_len = 1
                while s < 12 and ring[pp + s] == p2:
                    s += 3
                    s3_c_len += 1
                if s3_a_len < s3_c_len:
                    s3_c_len = s3_a_len

            s3_alt_len = 1
            if s3_b_len > 1:
                s3_alt_len = s3_b_len
                if s3_c_len <= s3_b_len:
                    s3_alt_len = s3_c_len

            if s1_len == 12:
                mode = 0
                has_pattern = True
                count = 12
            elif s2_b_len == 6:
                mode = 2
                has_pattern = True
                count = 6
            elif s2_half == 6:
                mode = 1
                has_pattern = True
                count = 6
            elif s3_alt_len == 4:
                mode = 5
                has_pattern = True
                count = 4
            elif s3_b_len == 4:
                mode = 4
                has_pattern = True
                count = 4
            elif s3_c_len == 4:
                # Two literals instead of a mode-3 pattern, as in fine_encoder.
                dst[o] = ring[start]
                dst[o + 1] = ring[start + 1]
                o += 2
                start += 2
            elif s3_a_len == 4:
                mode = 3
                has_pattern = True
                count = 4
            else:
                literal = False
                lit_count = 2
                if s3_alt_len >= 2:
                    lit_count = s3_alt_len
                    if s3_alt_len > 2:
                        mode = 5
                        count = s3_alt_len
                        do_flush = True
                    else:
                        lit_count = s3_alt_len * 3
                        literal = True
                elif s3_b_len > 1:
                    lit_count = s3_b_len
                    if s3_b_len > 2:
                        mode = 4
                        count = s3_b_len
                        do_flush = True
                    else:
                        lit_count = s3_b_len * 3
                        literal = True
                elif s3_c_len >= 2:
                    lit_count = 2
                    literal = True
                elif s3_a_len >= 3:
                    lit_count = s3_a_len
                    # Unreachable, as in fine_encoder.
                    if s3_a_len > 3:
                        mode = 3
                        count = s3_a_len
                        do_flush = True
                    else:
                        lit_count = s3_a_len * 3
                        literal = True
                elif s2_b_len >= 2:
                    lit_count = s2_b_len
                    if s2_b_len > 4:
                        mode = 2
                        count = s2_b_len
                        do_flush = True
                    else:
                        lit_count = lit_count << 1
                        literal = True
                elif s2_half >= 3:
                    lit_count = s2_half
                    if s2_half > 3:
                        mode = 1
                        count = s2_half
                        do_flush = True
                    else:
                        lit_count = lit_count << 1
                        literal = True
                else:
                    if s1_len < 3:
                        lit_count = 1
                    else:
                        lit_count = s1_len
                        # Unreachable, as in fine_encoder.
                        if s1_len > 5:
                            mode = 0
                            count = s1_len
                            do_flush = True
                    if not do_flush:
                        literal = True
                if literal:
                    for k in range(lit_count):
                        dst[o] = ring[start]
                        o += 1
                        start += 1

        if do_flush:
            start = _pattern_flush(dst, &o, mode, count, p0, p1, p2, ring, start)
            has_pattern = False
        if not skip_phase_reset:
            phase = 0

    if has_pattern:
        start = _pattern_flush(dst, &o, mode, count, p0, p1, p2, ring, start)
    while start < n:
        dst[o] = ring[start]
        o += 1
        start += 1
    return o


cdef Py_ssize_t encode_fine_plane_c(const uchar *src, Py_ssize_t n, uchar *scratch, uchar *dst) noexcept nogil:
    """Both stages; `scratch` and `dst` must each hold 2 * n + 16 bytes.

    Returns:
        Number of bytes written to `dst`; 0 for an all-zero scanline.
    """
    cdef Py_ssize_t i, m
    for i in range(n):
        if src[i]:
            break
    else:
        return 0
    m = rle_preencode_c(src, n, scratch)
    return pattern_encode_c(scratch, m, dst)


ctypedef Py_ssize_t (*_stage_fn)(const uchar *, Py_ssize_t, uchar *) noexcept nogil


cdef bytes _run_stage(_stage_fn fn, const uchar[::1] data):
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t m
    if n == 0:
        return b""
    cdef uchar *dst = <uchar *>malloc(2 * n + 16)
    if dst == NULL:
        raise MemoryError()
    try:
        with nogil:
            m = fn(&data[0], n, dst)
        return PyBytes_FromStringAndSize(<char *>dst, m)
    finally:
        free(dst)


def rle_preencode(const uchar[::1] data):
    """Stage 1 of the Fine codec (fine_encoder.compress_rle_preencode).

    Returns:
        Stage-1 byte stream using opcodes 0x10..0xFF.
    """
    return _run_stage(rle_preencode_c, data)


def pattern_encode(const uchar[::1] data):
    """Stage 2 of the Fine codec (fine_encoder.compress_pattern_encode).

    Returns:
        Final compressed byte stream including pattern headers.
    """
    return _run_stage(pattern_encode_c, data)


def encode_fine_plane(const uchar[::1] data):
    """Encode one Fine scanline through both stages (fine_encoder.encode_fine_plane).

    Returns:
        Compressed bytes, or empty bytes if the scanline is all-zero.
    """
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t m
    if n == 0:
        return b""
    cdef uchar *buf = <uchar *>malloc(2 * (2 * n + 16))
    if buf == NULL:
        raise MemoryError()
    try:
        with nogil:
            m = encode_fine_plane_c(&data[0], n, buf, buf + 2 * n + 16)
        return PyBytes_FromStringAndSize(<char *>(buf + 2 * n + 16), m)
    finally:
        free(buf)


def dither_row_4bpp(const uchar[:] row, const uchar[:, :, :] levels, Py_ssize_t y, Py_ssize_t width):
    """Nibble-table dither of one plane row: levels[255 - row[x], y % h, x % w], packed.

    Packs as dither._nibble_pack: the first pixel of each pair in the high
    nibble, a zero nibble after an odd last pixel.

    Returns:
        (width + 1) // 2 bytes.

    Raises:
        ValueError: If `row` is shorter than `width` or `levels` is not (256, h, w).
    """
    cdef Py_ssize_t h = levels.shape[1]
    cdef Py_ssize_t w = levels.shape[2]
    cdef Py_ssize_t x, col = 0
    cdef Py_ssize_t yy
    cdef uchar a
    if row.shape[0] < width or levels.shape[0] != 256 or h == 0 or w == 0 or width < 0:
        raise ValueError("row shorter than width or levels not of shape (256, h, w)")
    yy = y % h
    if yy < 0:
        yy += h
    cdef bytes out = PyBytes_FromStringAndSize(NULL, (width + 1) // 2)
    cdef uchar *po = <uchar *><char *>out
    with nogil:
        for x in range(width):
            a = levels[255 - row[x], yy, col]
            if x & 1:
                po[x >> 1] |= a
            else:
                # uint8 shift as in _nibble_pack: bits above the nibble fall off.
                po[x >> 1] = <uchar>(a << 4)
            col += 1
            if col == w:
                col = 0
    return out


_LUT_SIZE = 256 * 256 * 256 * 4


cdef inline void _pixel_kcmy(
    const uchar *px, bint has_lut, const uchar *lut, const int *grid, const uchar *weights, const int *black, uchar *kcmy
) noexcept nogil:
    cdef size_t off
    if has_lut:
        off = ((<size_t>px[0] << 16) | (<size_t>px[1] << 8) | px[2]) << 2
        kcmy[0] = lut[off]
        kcmy[1] = lut[off + 1]
        kcmy[2] = lut[off + 2]
        kcmy[3] = lut[off + 3]
    else:
        interp_pixel(grid, weights, black, px[0], px[1], px[2], kcmy)


def render_band_fine(
    const unsigned char[:, :] rows,
    const unsigned char[::1] skip,
    const unsigned char[::1] overrun,
    Py_ssize_t width,
    Py_ssize_t first_line,
    Py_ssize_t sw,
    const unsigned char[::1] lut,
    const int[::1] grid,
    const unsigned char[::1] weights,
    const int[::1] black,
    const unsigned char[:, :, ::1] lv_k,
    const unsigned char[:, :, ::1] lv_c,
    const unsigned char[:, :, ::1] lv_m,
    const unsigned char[:, :, ::1] lv_y,
    int[:, ::1] lengths,
):
    """Colour-convert, 4bpp-dither and Fine-encode `rows` (page lines from `first_line`).

    Byte-identical to lines.encode_lines with is_fine=True.

    Args:
        rows: (n, >= width * 3) RGB scanlines; each row contiguous, any row
            stride (cropped or reversed views are fine).
        skip: n flags; a set flag emits a blank line without looking at it.
        overrun: n flags, the RGB part of lines.fine_overrun_planes (image
            geometry and the last two input pixels equal); the kernel adds
            the no-ink test of the last pixel and then clears byte 0 of the
            K, M and Y rows.
        width: pixels per input row.
        first_line: page line index of rows[0]; selects the dither rows.
        sw: image width in pixels; pixels past `width` count as white.
        lut: flat (256*256*256*4) KCMY inverse LUT, or empty to interpolate
            `grid` instead.
        grid, weights, black: colour grid, interpolation weights and black
            ink from `color_lut.interp_arrays`; used when `lut` is empty.
        lv_k, lv_c, lv_m, lv_y: per-channel nibble tables (256, h, w),
            `DitherChannel.nibble_levels`.
        lengths: (n, 4) output, encoded length per line and plane (K, C, M, Y);
            0 means the plane line has no ink.

    Returns:
        The encoded plane lines back to back, in line then plane order.

    Raises:
        ValueError: On inconsistent shapes.
        MemoryError: If a buffer cannot be allocated.
    """
    cdef Py_ssize_t n = rows.shape[0]
    cdef Py_ssize_t npx = width if width < sw else sw
    cdef Py_ssize_t bpl = (sw + 1) // 2
    cdef Py_ssize_t bound = 2 * bpl + 16
    cdef bint has_lut = lut.shape[0] != 0
    cdef Py_ssize_t i, x, line, pid, m
    cdef Py_ssize_t col[4]
    cdef Py_ssize_t lw[4]
    cdef Py_ssize_t lplane[4]
    cdef const uchar *lrow[4]
    cdef const uchar *lv[4]
    cdef uchar kcmy[4]
    cdef uchar a
    cdef const uchar *row
    cdef const int *pgrid = NULL
    cdef const uchar *pweights = NULL
    cdef const int *pblack = NULL
    cdef const uchar *plut = NULL
    cdef uchar *planes
    cdef uchar *scratch
    cdef uchar *out = NULL
    cdef uchar *grown
    cdef Py_ssize_t out_len = 0, out_cap = 0
    cdef bint failed = False

    if skip.shape[0] != n or overrun.shape[0] != n or lengths.shape[0] != n or lengths.shape[1] != 4:
        raise ValueError("skip, overrun and lengths must have one entry per row")
    if n and rows.shape[1] < width * 3:
        raise ValueError("rows shorter than width * 3")
    if rows.shape[1] > 1 and rows.strides[1] != 1:
        raise ValueError("rows must be contiguous scanlines")
    if has_lut and lut.shape[0] != _LUT_SIZE:
        raise ValueError("lut must be the flat 256*256*256*4 inverse LUT")
    if not has_lut and (grid.shape[0] != 4913 * 4 or weights.shape[0] != 17 * 289 * 9 or black.shape[0] != 4):
        raise ValueError("without an inverse LUT, grid, weights and black ink are required")
    if (
        lv_k.shape[0] != 256
        or lv_c.shape[0] != 256
        or lv_m.shape[0] != 256
        or lv_y.shape[0] != 256
        or min(lv_k.shape[1], lv_c.shape[1], lv_m.shape[1], lv_y.shape[1]) == 0
        or min(lv_k.shape[2], lv_c.shape[2], lv_m.shape[2], lv_y.shape[2]) == 0
    ):
        raise ValueError("nibble tables must be of shape (256, h, w)")
    if sw <= 0 or width < 0:
        raise ValueError("sw must be positive")

    if has_lut:
        plut = &lut[0]
    else:
        pgrid = &grid[0]
        pweights = &weights[0]
        pblack = &black[0]
    lv[0] = &lv_k[0, 0, 0]
    lv[1] = &lv_c[0, 0, 0]
    lv[2] = &lv_m[0, 0, 0]
    lv[3] = &lv_y[0, 0, 0]
    lw[0] = lv_k.shape[2]
    lw[1] = lv_c.shape[2]
    lw[2] = lv_m.shape[2]
    lw[3] = lv_y.shape[2]
    lplane[0] = lv_k.shape[1] * lw[0]
    lplane[1] = lv_c.shape[1] * lw[1]
    lplane[2] = lv_m.shape[1] * lw[2]
    lplane[3] = lv_y.shape[1] * lw[3]

    planes = <uchar *>malloc(4 * bpl)
    scratch = <uchar *>malloc(bound)
    failed = planes == NULL or scratch == NULL

    try:
        with nogil:
            for i in range(n):
                if failed:
                    break
                if skip[i]:
                    for pid in range(4):
                        lengths[i, pid] = 0
                    continue
                line = first_line + i
                row = &rows[i, 0]
                # Row of each channel's level-0 table at this line; level v is lplane * v further.
                lrow[0] = lv[0] + (line % lv_k.shape[1]) * lw[0]
                lrow[1] = lv[1] + (line % lv_c.shape[1]) * lw[1]
                lrow[2] = lv[2] + (line % lv_m.shape[1]) * lw[2]
                lrow[3] = lv[3] + (line % lv_y.shape[1]) * lw[3]
                memset(planes, 0, 4 * bpl)
                for pid in range(4):
                    col[pid] = 0
                for x in range(sw):
                    if x < npx:
                        _pixel_kcmy(row + 3 * x, has_lut, plut, pgrid, pweights, pblack, kcmy)
                    else:
                        kcmy[0] = 255
                        kcmy[1] = 255
                        kcmy[2] = 255
                        kcmy[3] = 255
                    for pid in range(4):
                        a = lrow[pid][(255 - kcmy[pid]) * lplane[pid] + col[pid]]
                        if x & 1:
                            planes[pid * bpl + (x >> 1)] |= a
                        else:
                            planes[pid * bpl + (x >> 1)] = <uchar>(a << 4)
                        col[pid] += 1
                        if col[pid] == lw[pid]:
                            col[pid] = 0
                if overrun[i] and width >= 1:
                    _pixel_kcmy(row + 3 * (width - 1), has_lut, plut, pgrid, pweights, pblack, kcmy)
                    if kcmy[0] == 255 and kcmy[1] == 255 and kcmy[2] == 255 and kcmy[3] == 255:
                        planes[0] = 0  # K
                        planes[2 * bpl] = 0  # M
                        planes[3 * bpl] = 0  # Y
                for pid in range(4):
                    if out_cap - out_len < bound:
                        out_cap = 2 * out_cap + 4 * bound
                        grown = <uchar *>realloc(out, out_cap)
                        if grown == NULL:
                            failed = True
                            break
                        out = grown
                    m = encode_fine_plane_c(planes + pid * bpl, bpl, scratch, out + out_len)
                    out_len += m
                    lengths[i, pid] = <int>m
        if failed:
            raise MemoryError()
        return PyBytes_FromStringAndSize(<char *>out, out_len)
    finally:
        free(planes)
        free(scratch)
        free(out)
