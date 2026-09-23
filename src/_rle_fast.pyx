# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Bit-packing helpers for the XL2HB plane encoders.

`group_bits` reads a byte stream as N-bit groups (MSB first) and
`pack_groups` is its inverse. Both are bit-buffer state machines that
benefit substantially from native compilation.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize


def group_bits(bytes data, int group_size):
    """Group raw bytes into MSB-first values of `group_size` bits each.

    Last group is zero-padded if the bit count isn't a multiple of
    `group_size`.

    Returns:
        list[int] of length ceil(len(data)*8 / group_size).
    """
    cdef Py_ssize_t n = len(data)
    cdef Py_ssize_t total_bits = n * 8
    if total_bits == 0:
        return []

    cdef Py_ssize_t n_full = total_bits // group_size
    cdef Py_ssize_t remaining = total_bits - n_full * group_size
    cdef Py_ssize_t total = n_full + (1 if remaining > 0 else 0)
    cdef const unsigned char *p = data
    cdef unsigned long long buf = 0
    cdef int buf_bits = 0
    cdef Py_ssize_t i, byte_i = 0
    cdef unsigned long long mask = (<unsigned long long>1 << group_size) - 1

    cdef list out = [0] * total

    for i in range(n_full):
        while buf_bits < group_size:
            buf = (buf << 8) | p[byte_i]
            byte_i += 1
            buf_bits += 8
        out[i] = <object>(<unsigned long long>((buf >> (buf_bits - group_size)) & mask))
        buf_bits -= group_size

    if remaining > 0:
        while byte_i < n:
            buf = (buf << 8) | p[byte_i]
            byte_i += 1
            buf_bits += 8
        # Take the next `remaining` bits MSB-first and place them at the
        # high end of a `group_size`-bit value (zero-padded at the bottom).
        out[n_full] = <object>(
            <unsigned long long>(
                ((buf >> (buf_bits - remaining)) & ((<unsigned long long>1 << remaining) - 1))
                << (group_size - remaining)
            )
        )

    return out


def pack_groups(list groups, int group_size):
    """Pack a list of N-bit values into bytes, MSB-first.

    Returns:
        bytes of length ceil(len(groups) * group_size / 8).
    """
    cdef Py_ssize_t n = len(groups)
    if n == 0:
        return b""

    cdef Py_ssize_t total_bits = n * group_size
    cdef Py_ssize_t out_len = (total_bits + 7) // 8
    cdef bytes out_obj = PyBytes_FromStringAndSize(NULL, out_len)
    cdef unsigned char *po = <unsigned char *><char *>out_obj

    cdef unsigned long long buf = 0
    cdef int buf_bits = 0
    cdef Py_ssize_t i, oi = 0
    cdef unsigned long long val
    cdef unsigned long long mask = (<unsigned long long>1 << group_size) - 1

    for i in range(n):
        val = <unsigned long long>(<object>groups[i]) & mask
        buf = (buf << group_size) | val
        buf_bits += group_size
        while buf_bits >= 8:
            po[oi] = <unsigned char>((buf >> (buf_bits - 8)) & 0xFF)
            oi += 1
            buf_bits -= 8

    if buf_bits > 0:
        po[oi] = <unsigned char>((buf << (8 - buf_bits)) & 0xFF)

    return out_obj


# ---------------------------------------------------------------------------
# Sliding-window RLE encoder (native port of rle.sw_rle_encode + plane glue)
# ---------------------------------------------------------------------------

cdef bint _ensure(OutBuf *o, Py_ssize_t extra) noexcept nogil:
    """Grow `o` to hold `extra` more bytes; on allocation failure set `o.failed`."""
    cdef Py_ssize_t need = o.len + extra
    cdef Py_ssize_t cap
    cdef unsigned char *p
    if o.failed:
        return False
    if need <= o.cap:
        return True
    cap = o.cap * 2
    if cap < need:
        cap = need
    p = <unsigned char *>realloc(o.data, cap)
    if p == NULL:
        o.failed = True
        return False
    o.data = p
    o.cap = cap
    return True


cdef inline void _put(OutBuf *o, unsigned char b) noexcept nogil:
    if o.len >= o.cap and not _ensure(o, 1):
        return
    o.data[o.len] = b
    o.len += 1


cdef void _count_ext(OutBuf *o, Py_ssize_t remaining) noexcept nogil:
    while remaining > 254:
        _put(o, 0xFF)
        remaining -= 255
    _put(o, <unsigned char>(remaining & 0xFF))


cdef void _emit_run(OutBuf *o, unsigned int value, Py_ssize_t count, int bits) noexcept nogil:
    cdef unsigned int v_hi
    if bits == 10:
        v_hi = (value >> 8) & 0x3
        if count <= 31:
            _put(o, <unsigned char>(0x80 | ((count - 1) << 2) | v_hi))
            _put(o, <unsigned char>(value & 0xFF))
        else:
            _put(o, <unsigned char>(0xFC | v_hi))
            _put(o, <unsigned char>(value & 0xFF))
            _count_ext(o, count - 32)
        return

    v_hi = (value >> (bits - 4)) & 0xF
    if count < 8:
        _put(o, <unsigned char>(0x80 | ((count - 1) << 4) | v_hi))
    else:
        _put(o, <unsigned char>(0xF0 | v_hi))
    if bits > 12:
        _put(o, <unsigned char>((value >> 8) & 0xFF))
    _put(o, <unsigned char>(value & 0xFF))
    if count >= 8:
        _count_ext(o, count - 8)


cdef void _emit_literal(OutBuf *o, const unsigned int *vals, Py_ssize_t count, int bits) noexcept nogil:
    cdef unsigned long long buf = 0
    cdef int buf_bits = 0
    cdef Py_ssize_t i
    if count < 0x41:
        _put(o, <unsigned char>((count - 2) | 0x40))
    else:
        _put(o, 0x7F)
        _count_ext(o, count - 0x41)
    if not _ensure(o, (count * bits + 7) // 8):
        return
    for i in range(count):
        buf = (buf << bits) | vals[i]
        buf_bits += bits
        while buf_bits >= 8:
            o.data[o.len] = <unsigned char>((buf >> (buf_bits - 8)) & 0xFF)
            o.len += 1
            buf_bits -= 8
    if buf_bits > 0:
        o.data[o.len] = <unsigned char>((buf << (8 - buf_bits)) & 0xFF)
        o.len += 1


cdef void _emit_context_skip(OutBuf *o, Py_ssize_t count) noexcept nogil:
    if count < 64:
        _put(o, <unsigned char>(count - 1))
    else:
        _put(o, 0x3F)
        _count_ext(o, count - 64)


cdef Py_ssize_t _extract_words(
    const unsigned char *p, Py_ssize_t n_bytes, int read_group, int encode_group, unsigned int *out
) noexcept nogil:
    """Split the scanline into `encode_group`-bit words, MSB first.

    Bits past the last full `read_group` are treated as zero, matching
    rle.data_to_encode_groups (read_group=1 gives plain group_bits).
    """
    cdef Py_ssize_t total_bits = n_bytes * 8
    cdef Py_ssize_t covered = (total_bits // read_group) * read_group
    cdef Py_ssize_t n_words = (total_bits + encode_group - 1) // encode_group
    cdef Py_ssize_t i, bitpos = 0, byte_i = 0, take
    cdef unsigned long long buf = 0
    cdef int buf_bits = 0
    cdef unsigned int w
    for i in range(n_words):
        # Refill so that at least encode_group bits are buffered (zeros past end).
        while buf_bits < encode_group:
            if byte_i < n_bytes:
                buf = (buf << 8) | p[byte_i]
            else:
                buf = buf << 8
            byte_i += 1
            buf_bits += 8
        w = <unsigned int>((buf >> (buf_bits - encode_group)) & ((1ULL << encode_group) - 1))
        buf_bits -= encode_group
        # Zero the bits of this word that lie at or beyond `covered`.
        if bitpos + encode_group > covered:
            take = covered - bitpos
            if take <= 0:
                w = 0
            else:
                w &= <unsigned int>(((1ULL << take) - 1) << (encode_group - take))
        bitpos += encode_group
        out[i] = w
    return n_words


cdef void _sw_rle(const unsigned int *words, Py_ssize_t n, int bits, OutBuf *o, unsigned int *wbuf) noexcept nogil:
    """Port of rle.sw_rle_encode; state names follow the Python version.

    `wbuf` is scratch space for at least n + 2 words.
    """
    cdef int ctx_size = 5 if bits == 10 else 3
    cdef Py_ssize_t lit_overflow = 0xFFF if bits == 10 else 0x7FF
    cdef bint run_break_to_ctx = bits == 10
    cdef unsigned int ctx[5]
    cdef Py_ssize_t wi = 0, match_count = 0, run_len = 0, lit_n, k
    cdef Py_ssize_t wlen = 0
    cdef unsigned int cur, ref, w, last
    cdef int state
    # States: 0 MAIN, 1 RUN, 2 LITERAL, 3 CONTEXT_SKIP, 4 FINALIZE_LIT, 5 DONE
    for k in range(5):
        ctx[k] = 0
    if n == 0:
        return
    cur = words[0]
    wi = 1
    ctx[ctx_size - 1] = cur
    if cur == 0:
        match_count = 1
    state = 0

    while True:
        if state == 0:  # MAIN
            ref = ctx[1]
            for k in range(ctx_size - 1):
                ctx[k] = ctx[k + 1]
            wbuf[0] = cur
            wlen = 1
            run_len = 1
            if wi >= n:
                state = 4
                continue
            w = words[wi]
            wi += 1
            if ref == w:
                match_count += 1
            else:
                match_count = 0
            ctx[ctx_size - 1] = w
            if cur == w:
                run_len = 2
                state = 1
                continue
            if match_count == 2:
                _emit_run(o, cur, 1, bits)
                state = 3
                continue
            wbuf[1] = w
            wlen = 2
            run_len = 2
            state = 2
            continue

        if state == 1:  # RUN
            ref = ctx[1]
            for k in range(ctx_size - 1):
                ctx[k] = ctx[k + 1]
            if wi >= n:
                _emit_run(o, cur, run_len, bits)
                return
            w = words[wi]
            wi += 1
            if ref == w:
                match_count += 1
            else:
                match_count = 0
            ctx[ctx_size - 1] = w
            if cur == w:
                run_len += 1
                continue
            _emit_run(o, cur, run_len, bits)
            cur = w
            if match_count != 0:
                match_count = 1
                if run_break_to_ctx:
                    state = 3
                    continue
            state = 0
            continue

        if state == 2:  # LITERAL
            ref = ctx[1]
            for k in range(ctx_size - 1):
                ctx[k] = ctx[k + 1]
            if wi >= n:
                state = 4
                continue
            w = words[wi]
            wi += 1
            if ref == w:
                match_count += 1
            else:
                match_count = 0
            ctx[ctx_size - 1] = w

            if match_count > 1:
                if match_count - 1 < run_len:
                    lit_n = run_len - match_count + 1
                    if lit_n == 1:
                        _emit_run(o, wbuf[0], 1, bits)
                    else:
                        _emit_literal(o, wbuf, lit_n, bits)
                    state = 3
                    continue
                _emit_run(o, wbuf[0], 1, bits)
                state = 3
                continue

            if wbuf[wlen - 1] == w:
                if run_len > 2:
                    _emit_literal(o, wbuf, wlen - 1, bits)
                elif run_len == 2:
                    _emit_run(o, wbuf[0], 1, bits)
                cur = w
                wbuf[0] = w
                wlen = 1
                run_len = 2
                if match_count > 2:
                    match_count = 2
                state = 1
                continue

            if run_len + 1 > lit_overflow:
                lit_n = run_len - 1
                if lit_n == 1:
                    _emit_run(o, wbuf[0], 1, bits)
                else:
                    _emit_literal(o, wbuf, lit_n, bits)
                last = wbuf[wlen - 1]
                _emit_run(o, last, 1, bits)
                cur = w
                wbuf[0] = last
                wbuf[1] = w
                wlen = 2
                run_len = 2
                if match_count > 2:
                    match_count = 2
                continue

            wbuf[wlen] = w
            wlen += 1
            run_len += 1
            continue

        if state == 3:  # CONTEXT_SKIP
            ref = ctx[1]
            for k in range(ctx_size - 1):
                ctx[k] = ctx[k + 1]
            if wi >= n:
                _emit_context_skip(o, match_count)
                return
            w = words[wi]
            wi += 1
            ctx[ctx_size - 1] = w
            if ref != w:
                _emit_context_skip(o, match_count)
                match_count = 0
                cur = w
                state = 0
                continue
            match_count += 1
            continue

        # state == 4: FINALIZE_LIT
        if match_count < run_len:
            lit_n = run_len - match_count
            if lit_n == 1:
                _emit_run(o, wbuf[0], 1, bits)
            else:
                _emit_literal(o, wbuf, lit_n, bits)
        else:
            _emit_run(o, wbuf[0], 1, bits)
        if match_count > 0:
            _emit_context_skip(o, match_count)
        return


cdef Py_ssize_t encode_line(
    const unsigned char *p,
    Py_ssize_t n_bytes,
    int read_group,
    int encode_group,
    int bits,
    unsigned int *scratch,
    OutBuf *o,
) noexcept nogil:
    """Append one plane scanline, sliding-window RLE encoded, to `o`.

    Native equivalent of plane_encoders._encode_via_sw_rle over the words
    from rle.data_to_encode_groups(data, read_group, encode_group)
    (read_group=1 selects plain group_bits). `bits` picks the config:
    12 (CONFIG_12BIT), 20 (CONFIG_20BIT) or 10 (CONFIG_10BIT). `scratch`
    must hold encode_scratch_words(n_bytes, encode_group) words.

    Returns the number of bytes appended (0 if every word is zero), or -1
    if `o` could not grow (`o.failed` is set).
    """
    cdef Py_ssize_t n_words, i, pad_count, padded_bytes
    cdef Py_ssize_t start = o.len
    cdef unsigned int *words = scratch
    cdef bint any_ink = False

    if n_bytes == 0:
        return 0
    n_words = _extract_words(p, n_bytes, read_group, encode_group, words)
    for i in range(n_words):
        if words[i] != 0:
            any_ink = True
            break
    if not any_ink:
        return 0

    _ensure(o, n_words * 3 + 64)
    _sw_rle(words, n_words, bits, o, scratch + n_words)

    if o.len - start > n_bytes + 0x14:
        # Raw fallback, see rle.finalize_compressed.
        o.len = start
        if n_words < 0x41:
            _put(o, <unsigned char>((n_words - 2) | 0x40))
        else:
            _put(o, 0x7F)
            _count_ext(o, n_words - 0x41)
        padded_bytes = (n_words * encode_group + 7) // 8
        pad_count = padded_bytes - n_bytes
        if _ensure(o, n_bytes + pad_count):
            memcpy(o.data + o.len, p, n_bytes)
            o.len += n_bytes
            for i in range(pad_count):
                o.data[o.len] = 0
                o.len += 1

    if o.failed:
        return -1
    return o.len - start


cdef Py_ssize_t encode_scratch_words(Py_ssize_t n_bytes, int encode_group) noexcept nogil:
    """Words of scratch space `encode_line` needs for an `n_bytes` scanline."""
    return 2 * ((n_bytes * 8 + encode_group - 1) // encode_group) + 2


def encode_sw_rle(bytes data, int read_group, int encode_group, int bits):
    """Encode one plane scanline with the sliding-window RLE, incl. raw fallback.

    See `encode_line`.

    Returns:
        Compressed bytes; empty if every word is zero.
    """
    cdef Py_ssize_t n_bytes = len(data)
    cdef const unsigned char *p = data
    cdef unsigned int *scratch
    cdef OutBuf o
    cdef Py_ssize_t n

    if bits != 10 and bits != 12 and bits != 20:
        raise ValueError(f"unsupported word size {bits}")
    if n_bytes == 0:
        return b""
    scratch = <unsigned int *>malloc(encode_scratch_words(n_bytes, encode_group) * sizeof(unsigned int))
    if scratch == NULL:
        raise MemoryError()
    o.data = NULL
    o.len = 0
    o.cap = 0
    o.failed = False
    try:
        n = encode_line(p, n_bytes, read_group, encode_group, bits, scratch, &o)
        if n < 0:
            raise MemoryError()
        return PyBytes_FromStringAndSize(<char *>o.data, o.len)
    finally:
        free(scratch)
        free(o.data)
