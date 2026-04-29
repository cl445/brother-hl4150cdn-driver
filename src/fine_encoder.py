"""Two-stage compression for Fine mode (1200 dpi).

* :func:`compress_rle_preencode` — Stage 1: nibble-aware RLE that maps
  4bpp scanlines to bytes in the range 0x10..0xFF.
* :func:`compress_jpegls_encode` — Stage 2: pattern-matching encoder
  that detects strides 1/2/3 over a 12-byte ring buffer; bytes 0x00..0x0F
  are reserved as pattern headers.
* :func:`encode_fine_plane` — runs both stages back-to-back; returns
  empty bytes for an all-zero scanline.
"""


def compress_rle_preencode(data: bytes) -> bytes:
    """Stage 1 of Fine mode compression: nibble-aware RLE pre-encoding.

    Input: 4bpp nibble-packed scanline data (e.g. 2480 bytes for Fine A4).
    Quantizes each nibble: if != 0xF, clear LSB (16 levels -> 9 levels).
    Tracks white (0x00) and black (0xFF) runs with custom encoding.

    Output codes 0x10-0xFF (0x00-0x0F reserved for Stage 2).

    Returns:
        Stage-1 byte stream using opcodes 0x10..0xFF.
    """
    output = bytearray()
    run_white = 0
    run_black = 0

    def _flush_white_run() -> None:
        nonlocal run_white
        while run_white > 8:
            output.append(0x17)
            run_white -= 8
        output.append((run_white - 1) | 0x10)
        run_white = 0

    def _flush_black_run() -> None:
        nonlocal run_black
        while run_black > 8:
            output.append(0x1F)
            run_black -= 8
        output.append((run_black - 1) | 0x18)
        run_black = 0

    for byte_val in data:
        # Quantize: clear LSB of each nibble unless it's 0xF
        src_val = byte_val
        if (src_val & 0xF0) != 0xF0:
            src_val &= 0xEF
        if (src_val & 0x0F) != 0x0F:
            src_val &= 0xFE

        if src_val == 0x00:
            # White byte
            if run_black == 0:
                run_white += 1
            else:
                if run_black < 5:
                    output.append(((run_black * 8 - 8) & 0xFF) | 0x80)
                else:
                    _flush_black_run()
                    run_white += 1
                run_black = 0

        elif src_val == 0xFF:
            # Black byte
            if run_white == 0:
                run_black += 1
            else:
                if run_white < 3:
                    output.append(((run_white * 8 - 8) & 0xFF) | 0x27)
                else:
                    _flush_white_run()
                    run_black += 1
                run_white = 0

        else:
            # Mixed byte: hi_nibble and lo_nibble
            hi_nibble = src_val & 0xF0
            lo_nibble = src_val & 0x0F

            if hi_nibble == 0xF0:
                # Hi = full ink (0xF_)
                if run_white == 0:
                    if run_black != 0:
                        if run_black > 3:
                            run_black -= 3
                            _flush_black_run()
                            run_black = 3
                        output.append((run_black * 8) | (lo_nibble >> 1) | 0xC0)
                        run_black = 0
                        continue
                else:
                    if lo_nibble == 0:
                        if run_white > 3:
                            run_white -= 3
                            _flush_white_run()
                            run_white = 3
                        output.append((run_white * 8) | 0xA7)
                        run_white = 0
                        continue
                    _flush_white_run()
                output.append((lo_nibble >> 1) | 0xC0)

            elif lo_nibble == 0:
                # Lo = no ink (0x_0)
                if run_white != 0:
                    if run_white > 3:
                        run_white -= 3
                        _flush_white_run()
                        run_white = 3
                    output.append(((run_white * 8 - 8) & 0xFF) | ((hi_nibble >> 5) - 1) | 0xA0)
                    run_white = 0
                    continue
                if run_black != 0:
                    if run_black > 4:
                        run_black -= 4
                        _flush_black_run()
                        run_black = 4
                    output.append(((run_black * 8 - 8) & 0xFF) | (hi_nibble >> 5) | 0x80)
                    run_black = 0
                    continue
                output.append(((hi_nibble >> 5) - 1) | 0xA0)

            elif lo_nibble == 0x0F:
                # Lo = full ink (0x_F)
                if run_white != 0:
                    if run_white > 2:
                        run_white -= 2
                        _flush_white_run()
                        run_white = 2
                    if hi_nibble == 0:
                        output.append(((run_white * 8 - 8) & 0xFF) | 0x37)
                    else:
                        output.append(((run_white * 8 - 8) & 0xFF) | ((hi_nibble >> 5) - 1) | 0x20)
                    run_white = 0
                    continue
                if run_black != 0:
                    if run_black > 4:
                        run_black -= 4
                        _flush_black_run()
                        run_black = 4
                    output.append(((run_black * 8 - 8) & 0xFF) | (hi_nibble >> 5) | 0xE0)
                    run_black = 0
                    continue
                output.append((hi_nibble >> 2) | 0x47)

            # General mixed: both nibbles non-zero, non-full
            elif hi_nibble == 0:
                if run_black == 0:
                    if run_white != 0:
                        if run_white > 2:
                            run_white -= 2
                            _flush_white_run()
                            run_white = 2
                        output.append(((run_white * 8 - 8) & 0xFF) | ((lo_nibble >> 1) - 1) | 0x30)
                        run_white = 0
                        continue
                else:
                    _flush_black_run()
                output.append(((lo_nibble >> 1) - 1) | 0x40)
            else:
                if run_white == 0:
                    if run_black != 0:
                        _flush_black_run()
                else:
                    _flush_white_run()
                output.append((hi_nibble >> 2) | ((lo_nibble >> 1) - 1) | 0x40)

    # Flush trailing runs
    if run_white != 0:
        _flush_white_run()
    elif run_black != 0:
        _flush_black_run()

    return bytes(output)


def _jpegls_emit_ext_count(output: bytearray, remaining: int) -> None:
    """Emit extended length bytes for the Stage 2 pattern encoder."""
    while remaining > 0xFE:
        output.append(0xFF)
        remaining -= 0xFF
    output.append(remaining & 0xFF)


def _jpegls_flush_pattern(
    output: bytearray,
    pattern_mode: int,
    match_count: int,
    pat_bytes: tuple[int, int, int],
    ring: bytearray,
    pattern_start: int,
) -> int:
    """Flush a detected pattern to output.

    Returns:
        Updated `pattern_start` index after consuming the matched bytes.
    """
    p0, p1, p2 = pat_bytes

    if pattern_mode == 0:
        # Stride-1 repeat (A,A,A,...)
        if match_count - 3 < 3:
            output.append(match_count - 3)
        else:
            output.append(3)
            _jpegls_emit_ext_count(output, match_count - 6)
        output.append(p0)
        pattern_start += match_count

    elif pattern_mode == 1:
        # Stride-2 half (A,x,A,x,...) — emit A then each odd byte
        if match_count == 3:
            output.append(0x08)
        else:
            output.append(0x09)
            _jpegls_emit_ext_count(output, match_count - 4)
        output.append(p0)
        for _ in range(match_count):
            output.append(ring[pattern_start + 1])
            pattern_start += 2

    elif pattern_mode == 2:
        # Stride-2 full (A,B,A,B,...)
        if match_count - 2 < 3:
            output.append((match_count - 2) | 0x04)
        else:
            output.append(0x07)
            _jpegls_emit_ext_count(output, match_count - 5)
        output.append(p0)
        output.append(p1)
        pattern_start += match_count * 2

    elif pattern_mode == 3:
        # Stride-3 third (A,x,x,A,x,x,...) — emit A then each non-A pair
        if match_count == 3:
            output.append(0x0A)
        else:
            output.append(0x0B)
            _jpegls_emit_ext_count(output, match_count - 4)
        output.append(p0)
        for _ in range(match_count):
            output.append(ring[pattern_start + 1])
            output.append(ring[pattern_start + 2])
            pattern_start += 3

    elif pattern_mode == 4:
        # Stride-3 two-thirds (A,B,x,A,B,x,...) — emit A,B then each third byte
        if match_count == 2:
            output.append(0x0C)
        else:
            output.append(0x0D)
            _jpegls_emit_ext_count(output, match_count - 3)
        output.append(p0)
        output.append(p1)
        for _ in range(match_count):
            output.append(ring[pattern_start + 2])
            pattern_start += 3

    elif pattern_mode == 5:
        # Stride-3 full (A,B,C,A,B,C,...)
        if match_count == 2:
            output.append(0x0E)
        else:
            output.append(0x0F)
            _jpegls_emit_ext_count(output, match_count - 3)
        output.append(p0)
        output.append(p1)
        output.append(p2)
        pattern_start += match_count * 3

    return pattern_start


def compress_jpegls_encode(data: bytes) -> bytes:
    """Stage 2 of Fine mode compression: pattern-matching encoder.

    Buffers 12 bytes, detects repeating patterns at strides 1/2/3.
    Bytes >= 0x10 are literal pass-through from Stage 1.
    Bytes 0x00-0x0F are pattern headers.

    Returns:
        Final compressed byte stream including pattern headers.
    """
    output = bytearray()

    ring = bytearray()
    pattern_start = 0

    has_pattern = False
    pattern_mode = 0
    match_count = 0
    phase_idx = 0
    pat_byte_0 = 0
    pat_byte_1 = 0
    pat_byte_2 = 0

    for src_val in data:
        ring.append(src_val)
        buf_len = len(ring) - pattern_start

        do_flush = False
        skip_phase_reset = False

        if has_pattern:
            phase_idx += 1
            pattern_continues = False

            if pattern_mode == 0:
                if src_val == pat_byte_0:
                    match_count += 1
                    phase_idx = 0
                    if buf_len < 0x400:
                        pattern_continues = True
            elif pattern_mode == 1:
                do_buffer_check = False
                if phase_idx == 1:
                    if src_val == pat_byte_0:
                        do_buffer_check = True
                else:
                    match_count += 1
                    phase_idx = 0
                    do_buffer_check = True
                if do_buffer_check and buf_len < 0x400:
                    pattern_continues = True
            elif pattern_mode == 2:
                do_buffer_check = False
                if phase_idx == 1:
                    if src_val == pat_byte_0:
                        do_buffer_check = True
                elif src_val == pat_byte_1:
                    match_count += 1
                    phase_idx = 0
                    do_buffer_check = True
                if do_buffer_check and buf_len < 0x400:
                    pattern_continues = True
            elif pattern_mode == 3:
                do_buffer_check = False
                if phase_idx == 1:
                    if src_val == pat_byte_0:
                        do_buffer_check = True
                else:
                    if phase_idx != 2:
                        match_count += 1
                        phase_idx = 0
                    do_buffer_check = True
                if do_buffer_check and buf_len < 0x400:
                    pattern_continues = True
            elif pattern_mode == 4:
                do_buffer_check = False
                if phase_idx == 1:
                    if src_val == pat_byte_0:
                        do_buffer_check = True
                elif phase_idx == 2:
                    if src_val == pat_byte_1:
                        do_buffer_check = True
                else:
                    match_count += 1
                    phase_idx = 0
                    do_buffer_check = True
                if do_buffer_check and buf_len < 0x400:
                    pattern_continues = True
            elif pattern_mode == 5:
                do_buffer_check = False
                if phase_idx == 1:
                    if src_val == pat_byte_0:
                        do_buffer_check = True
                elif phase_idx == 2:
                    if src_val == pat_byte_1:
                        do_buffer_check = True
                elif src_val == pat_byte_2:
                    match_count += 1
                    phase_idx = 0
                    do_buffer_check = True
                if do_buffer_check and buf_len < 0x400:
                    pattern_continues = True

            if pattern_continues:
                skip_phase_reset = True
            else:
                do_flush = True
                has_pattern = False
        elif buf_len != 12:
            skip_phase_reset = True
        else:
            # Analyze 12-byte window for patterns
            pp = pattern_start
            pat_byte_0 = ring[pp]

            # Stride-1: consecutive bytes matching pat_byte_0
            stride1_len = 1
            while stride1_len < 12 and ring[pp + stride1_len] == pat_byte_0:
                stride1_len += 1

            # Stride-2 half: pat_byte_0 at even positions
            stride2_half = 2
            while stride2_half < 12 and ring[pp + stride2_half] == pat_byte_0:
                stride2_half += 2
            stride2_half >>= 1

            # Stride-3 first: pat_byte_0 at positions 0,3,6,9
            stride3_first = 3
            stride2_len = 1
            while stride3_first < 12 and ring[pp + stride3_first] == pat_byte_0:
                stride3_first += 3
                stride2_len += 1

            pat_byte_1 = ring[pp + 1]

            # Stride-2 B: pat_byte_1 at odd positions (3,5,7,...)
            stride2_b_len = 1
            if stride2_half > 1:
                s = 3
                while s < 12 and ring[pp + s] == pat_byte_1:
                    s += 2
                stride2_b_len = (s - 1) >> 1
                stride2_b_len = min(stride2_b_len, stride2_half)

            pat_byte_2 = ring[pp + 2]

            # Stride-3 B and C lengths
            if stride2_len < 2:
                stride3_b_len = 1
                stride3_c_len = 1
            else:
                s = 4
                stride3_b_len = 1
                while s < 12 and ring[pp + s] == pat_byte_1:
                    s += 3
                    stride3_b_len += 1
                stride3_b_len = min(stride3_b_len, stride2_len)

                s = 5
                stride3_c_len = 1
                while s < 12 and ring[pp + s] == pat_byte_2:
                    s += 3
                    stride3_c_len += 1
                stride3_c_len = min(stride3_c_len, stride2_len)

            # Min of stride3_b and stride3_c
            stride3_alt_len = 1
            if stride3_b_len > 1:
                stride3_alt_len = stride3_b_len
                if stride3_c_len <= stride3_b_len:
                    stride3_alt_len = stride3_c_len

            # Select best pattern
            if stride1_len == 12:
                pattern_mode = 0
                has_pattern = True
                match_count = 12
            elif stride2_b_len == 6:
                pattern_mode = 2
                has_pattern = True
                match_count = 6
            elif stride2_half == 6:
                pattern_mode = 1
                has_pattern = True
                match_count = 6
            elif stride3_alt_len == 4:
                pattern_mode = 5
                has_pattern = True
                match_count = 4
            elif stride3_b_len == 4:
                pattern_mode = 4
                has_pattern = True
                match_count = 4
            elif stride2_len == 4:
                pattern_mode = 3
                has_pattern = True
                match_count = 4
            else:
                # No long pattern — try short flush or literal copy
                do_literal_copy = False
                lit_count = 2

                if stride3_c_len == 4 if stride2_len >= 2 else False:
                    do_literal_copy = True
                    lit_count = 2
                elif stride3_alt_len >= 2:
                    lit_count = stride3_alt_len
                    if stride3_alt_len > 2:
                        pattern_mode = 5
                        match_count = stride3_alt_len
                        do_flush = True
                    else:
                        lit_count = stride3_alt_len * 3
                        do_literal_copy = True
                elif stride3_b_len > 1:
                    lit_count = stride3_b_len
                    if stride3_b_len > 2:
                        pattern_mode = 4
                        match_count = stride3_b_len
                        do_flush = True
                    else:
                        lit_count = stride3_b_len * 3
                        do_literal_copy = True
                elif stride3_c_len >= 2 if stride2_len >= 2 else False:
                    lit_count = 2
                    do_literal_copy = True
                elif stride2_len >= 3:
                    lit_count = stride2_len
                    if stride2_len > 3:
                        pattern_mode = 3
                        match_count = stride2_len
                        do_flush = True
                    else:
                        lit_count = stride2_len * 3
                        do_literal_copy = True
                elif stride2_b_len >= 2:
                    lit_count = stride2_b_len
                    if stride2_b_len > 4:
                        pattern_mode = 2
                        match_count = stride2_b_len
                        do_flush = True
                    else:
                        lit_count = lit_count << 1
                        do_literal_copy = True
                elif stride2_half >= 3:
                    lit_count = stride2_half
                    if stride2_half > 3:
                        pattern_mode = 1
                        match_count = stride2_half
                        do_flush = True
                    else:
                        lit_count = lit_count << 1
                        do_literal_copy = True
                else:
                    if stride1_len < 3:
                        lit_count = 1
                    else:
                        lit_count = stride1_len
                        if stride1_len > 5:
                            pattern_mode = 0
                            match_count = stride1_len
                            do_flush = True
                    if not do_flush:
                        do_literal_copy = True

                if do_literal_copy:
                    for _ in range(lit_count):
                        output.append(ring[pattern_start])
                        pattern_start += 1
                        # has_pattern remains False

        if do_flush:
            pattern_start = _jpegls_flush_pattern(
                output,
                pattern_mode,
                match_count,
                (pat_byte_0, pat_byte_1, pat_byte_2),
                ring,
                pattern_start,
            )
            has_pattern = False

        if not skip_phase_reset:
            phase_idx = 0

    # End of input — flush remaining pattern + literals
    if has_pattern:
        pattern_start = _jpegls_flush_pattern(
            output,
            pattern_mode,
            match_count,
            (pat_byte_0, pat_byte_1, pat_byte_2),
            ring,
            pattern_start,
        )

    while pattern_start < len(ring):
        output.append(ring[pattern_start])
        pattern_start += 1

    return bytes(output)


def encode_fine_plane(data: bytes) -> bytes:
    """Encode a single Fine-mode scanline through both compression stages.

    Returns:
        Compressed bytes, or empty bytes if the scanline is all-zero.
    """
    if not any(data):
        return b""
    stage1 = compress_rle_preencode(data)
    if not stage1:
        return b""
    return compress_jpegls_encode(stage1)
