"""Bit-packing helpers and run-length encoders for the XL2HB plane data.

The plane encoders share three pieces of machinery:

* ``group_bits`` / ``pack_groups``: read raw scanline bytes as a stream
  of N-bit groups (12, 20, …) and pack groups back to bytes.
* ``_rle_encode``: simple RLE encoder used by the ``encode_m_plane_20``
  sub-block path.
* ``_sw_rle_encode``: unified sliding-window RLE encoder, parametrised
  by ``_SwRleConfig`` so a single state machine handles 12-bit, 20-bit
  and 10-bit variants.

``_finalize_compressed`` is the shared post-processing step: if the
compressed output exceeds the raw input by more than 0x14 bytes, fall
back to a literal dump of the raw scanline.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, auto

import numpy as np

try:
    from _rle_fast import group_bits, pack_groups  # type: ignore[import-not-found]

    HAS_CYTHON_RLE = True
except ImportError:
    HAS_CYTHON_RLE = False

    def group_bits(data: bytes, group_size: int) -> list[int]:
        """Group input byte data into values of group_size bits each (MSB first).

        Returns:
            List of integer groups; the last entry is zero-padded if the
            bit count is not a multiple of `group_size`.
        """
        total_bits = len(data) * 8
        if total_bits == 0:
            return []

        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        n_full = total_bits // group_size

        if n_full > 0:
            full_bits = bits[: n_full * group_size].reshape(n_full, group_size)
            powers = 1 << np.arange(group_size - 1, -1, -1, dtype=np.uint32)
            groups = (full_bits.astype(np.uint32) * powers).sum(axis=1).tolist()
        else:
            groups = []

        remaining = total_bits - n_full * group_size
        if remaining > 0:
            value = 0
            offset = n_full * group_size
            for i in range(remaining):
                if bits[offset + i]:
                    value |= 1 << (group_size - 1 - i)
            groups.append(value)

        return groups

    def pack_groups(groups: list[int], group_size: int) -> bytes:
        """Pack N-bit groups into bytes, MSB first.

        Returns:
            Packed byte string of length `ceil(len(groups) * group_size / 8)`.
        """
        if not groups:
            return b""

        arr = np.array(groups, dtype=np.uint32)
        shifts = np.arange(group_size - 1, -1, -1, dtype=np.uint32)
        bit_matrix = ((arr[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
        return np.packbits(bit_matrix.ravel()).tobytes()


def data_to_encode_groups(data: bytes, read_group: int, encode_group: int) -> list[int]:
    """Convert raw byte data to encode-size groups via the read-group process.

    1. Read floor(total_bits / read_group) full groups from input
    2. Regroup those bits into encode_group-size values
    3. Pad with zeros to ceil(total_bits / encode_group) total groups

    Returns:
        List of integer groups, padded to the encode-grid size.
    """
    total_bits = len(data) * 8
    n_read = total_bits // read_group
    bits_covered = n_read * read_group
    total_encode_groups = -(-total_bits // encode_group)  # ceil division

    covered_bytes = bits_covered // 8
    covered_remainder = bits_covered % 8

    if covered_remainder == 0:
        covered_data = data[:covered_bytes]
    else:
        buf = bytearray(data[:covered_bytes])
        mask = (0xFF << (8 - covered_remainder)) & 0xFF
        buf.append(data[covered_bytes] & mask)
        covered_data = bytes(buf)

    groups = group_bits(covered_data, encode_group)

    while len(groups) < total_encode_groups:
        groups.append(0)

    return groups


def emit_count_ext(output: bytearray, remaining: int) -> None:
    """Emit count-extension bytes: 0xFF for each 255, then a remainder byte."""
    while remaining > 254:
        output.append(0xFF)
        remaining -= 255
    output.append(remaining & 0xFF)


def emit_run(output: bytearray, value: int, count: int, value_bits: int = 12) -> None:
    """Emit a run of identical N-bit values.

    Args:
        output: Output buffer to append to.
        value: The N-bit value to repeat.
        count: Number of repetitions.
        value_bits: Size of each value in bits (12 or 20).
    """
    v_hi = (value >> (value_bits - 4)) & 0xF

    if count < 8:  # noqa: SIM108 — ternary unreadable with bitwise ops
        header = 0x80 | ((count - 1) << 4) | v_hi
    else:
        header = 0xF0 | v_hi

    if value_bits <= 12:
        output.extend((header, value & 0xFF))
    else:
        output.extend((header, (value >> 8) & 0xFF, value & 0xFF))

    if count >= 8:
        emit_count_ext(output, count - 8)


def emit_literal_block(output: bytearray, values: list[int], value_bits: int = 12) -> None:
    """Emit a block of literal N-bit values."""
    count = len(values)
    if count < 0x41:
        output.append((count - 2) | 0x40)
    else:
        output.append(0x7F)
        emit_count_ext(output, count - 0x41)

    output.extend(pack_groups(values, value_bits))


def rle_encode(groups: list[int], value_bits: int = 12) -> bytes:
    """RLE-encode a list of N-bit groups.

    Args:
        groups: List of N-bit values to encode.
        value_bits: Size of each value in bits (12 or 20).

    Returns:
        Compressed data bytes (empty if all-zero).
    """
    n = len(groups)
    if n == 0:
        return b""

    arr = np.array(groups, dtype=np.uint32)
    if not np.any(arr):
        return b""

    changes = np.empty(n, dtype=bool)
    changes[0] = True
    changes[1:] = arr[1:] != arr[:-1]
    starts = np.where(changes)[0]
    nr = len(starts)
    rlens = np.empty(nr, dtype=np.intp)
    rlens[:-1] = starts[1:] - starts[:-1]
    rlens[-1] = n - starts[-1]
    rvals = arr[starts].tolist()
    rlens = rlens.tolist()

    output = bytearray()
    ri = 0

    while ri < nr:
        lit_start = ri
        while ri < nr and rlens[ri] == 1:
            ri += 1

        n_lit = ri - lit_start
        if n_lit > 0:
            if n_lit == 1:
                v = rvals[lit_start]
                if value_bits <= 12:
                    output.extend((0x80 | ((v >> 8) & 0xF), v & 0xFF))
                else:
                    output.extend((0x80 | ((v >> 16) & 0xF), (v >> 8) & 0xFF, v & 0xFF))
            else:
                emit_literal_block(output, rvals[lit_start:ri], value_bits)

        if ri >= nr:
            break

        emit_run(output, rvals[ri], rlens[ri], value_bits)
        ri += 1

    return bytes(output)


def emit_context_skip(output: bytearray, count: int) -> None:
    """Emit a context-skip command for the sliding-window encoder.

    Encodes a run of context-predicted words.

    Short form (count 1-63): ``[count - 1]`` (byte range 0x00-0x3E).
    Long form (count >= 64):  ``[0x3F] [count_ext...]``.
    """
    if count < 64:
        output.append(count - 1)
    else:
        output.append(0x3F)
        emit_count_ext(output, count - 64)


# ---------------------------------------------------------------------------
# Sliding-window RLE encoder (12-bit / 20-bit / 10-bit variants)
# ---------------------------------------------------------------------------


class _State(IntEnum):
    MAIN = auto()
    RUN = auto()
    LITERAL = auto()
    CONTEXT_SKIP = auto()
    FINALIZE_LIT = auto()
    DONE = auto()


class _ContextWindow3:
    """3-word context window for C/Y/K planes (prev[0] always 0)."""

    __slots__ = ("prev1", "prev2")

    def __init__(self) -> None:
        self.prev1 = 0
        self.prev2 = 0

    def store_first(self, w: int) -> None:
        self.prev2 = w

    def shift_and_get_ref(self) -> int:
        old = self.prev1
        self.prev1 = self.prev2
        return old

    def store(self, w: int) -> None:
        self.prev2 = w


class _ContextWindow5:
    """5-word context window for the M plane."""

    __slots__ = ("prev",)

    def __init__(self) -> None:
        self.prev = [0, 0, 0, 0, 0]

    def store_first(self, w: int) -> None:
        self.prev[4] = w

    def shift_and_get_ref(self) -> int:
        p = self.prev
        ref = p[1]
        p[0] = p[1]
        p[1] = p[2]
        p[2] = p[3]
        p[3] = p[4]
        return ref

    def store(self, w: int) -> None:
        self.prev[4] = w


@dataclass(frozen=True, slots=True)
class _SwRleConfig:
    make_context: Callable[[], _ContextWindow3 | _ContextWindow5]
    emit_run: Callable[[bytearray, int, int], None]
    emit_literal: Callable[[bytearray, list[int]], None]
    eof: int
    literal_overflow: int
    run_break_to_context_skip: bool


def _make_emit_run_nbit(value_bits: int) -> Callable[[bytearray, int, int], None]:
    def emit(output: bytearray, value: int, count: int) -> None:
        emit_run(output, value, count, value_bits=value_bits)

    return emit


def _make_emit_literal_nbit(value_bits: int) -> Callable[[bytearray, list[int]], None]:
    def emit(output: bytearray, values: list[int]) -> None:
        emit_literal_block(output, values, value_bits=value_bits)

    return emit


def _emit_run_10bit(output: bytearray, value: int, count: int) -> None:
    """Emit a run of identical 10-bit values using the 10-bit RLE format."""
    v_hi = (value >> 8) & 0x3
    v_lo = value & 0xFF
    if count <= 31:
        output.extend((0x80 | ((count - 1) << 2) | v_hi, v_lo))
    else:
        output.extend((0xFC | v_hi, v_lo))
        emit_count_ext(output, count - 32)


def _emit_literal_block_10bit(output: bytearray, values: list[int]) -> None:
    """Emit a literal block of 10-bit values."""
    count = len(values)
    if count < 65:
        output.append((count - 2) | 0x40)
    else:
        output.append(0x7F)
        emit_count_ext(output, count - 65)
    output.extend(pack_groups(values, 10))


CONFIG_12BIT = _SwRleConfig(
    make_context=_ContextWindow3,
    emit_run=_make_emit_run_nbit(12),
    emit_literal=_make_emit_literal_nbit(12),
    eof=0xFFFFFFFF,
    literal_overflow=0x7FF,
    run_break_to_context_skip=False,
)

CONFIG_20BIT = _SwRleConfig(
    make_context=_ContextWindow3,
    emit_run=_make_emit_run_nbit(20),
    emit_literal=_make_emit_literal_nbit(20),
    eof=0xFFFFFFFF,
    literal_overflow=0x7FF,
    run_break_to_context_skip=False,
)

CONFIG_10BIT = _SwRleConfig(
    make_context=_ContextWindow5,
    emit_run=_emit_run_10bit,
    emit_literal=_emit_literal_block_10bit,
    eof=0xFFFF,
    literal_overflow=0xFFF,
    run_break_to_context_skip=True,
)


def sw_rle_encode(words: list[int], cfg: _SwRleConfig) -> bytes:
    """Sliding-window RLE encoder, parametrised by `_SwRleConfig`.

    Returns:
        Encoded byte stream produced by the configured emit callbacks.
    """
    eof = cfg.eof
    emit_run_fn = cfg.emit_run
    emit_literal_fn = cfg.emit_literal
    lit_overflow = cfg.literal_overflow
    run_break_to_ctx = cfg.run_break_to_context_skip

    n = len(words)
    output = bytearray()
    ctx = cfg.make_context()
    match_count = 0

    wi = 0
    word_buf: list[int] = []
    run_len = 0

    def read_word() -> int:
        nonlocal wi
        if wi >= n:
            return eof
        w = words[wi]
        wi += 1
        return w

    w0 = read_word()
    if w0 == eof:
        return b""
    word_buf = [w0]
    ctx.store_first(w0)

    if w0 == 0:
        match_count = 1

    state = _State.MAIN
    cur = w0

    while True:
        if state == _State.MAIN:
            ref = ctx.shift_and_get_ref()
            word_buf = [cur]
            run_len = 1

            w1 = read_word()
            if w1 == eof:
                state = _State.FINALIZE_LIT
                continue

            if ref == w1:
                match_count += 1
            else:
                match_count = 0
            ctx.store(w1)

            if cur == w1:
                run_len = 2
                state = _State.RUN
                continue

            if match_count == 2:
                emit_run_fn(output, cur, 1)
                state = _State.CONTEXT_SKIP
                continue

            word_buf.append(w1)
            run_len = 2
            state = _State.LITERAL
            continue

        if state == _State.RUN:
            ref = ctx.shift_and_get_ref()
            w = read_word()
            if w == eof:
                emit_run_fn(output, cur, run_len)
                state = _State.DONE
                continue
            if ref == w:
                match_count += 1
            else:
                match_count = 0
            ctx.store(w)
            if cur == w:
                run_len += 1
                continue
            emit_run_fn(output, cur, run_len)
            cur = w
            if match_count != 0:
                match_count = 1
                if run_break_to_ctx:
                    state = _State.CONTEXT_SKIP
                    continue
            state = _State.MAIN
            continue

        if state == _State.LITERAL:
            ref = ctx.shift_and_get_ref()
            w = read_word()
            if w == eof:
                state = _State.FINALIZE_LIT
                continue

            if ref == w:
                match_count += 1
            else:
                match_count = 0
            ctx.store(w)

            if match_count > 1:
                if match_count - 1 < run_len:
                    lit_n = run_len - match_count + 1
                    if lit_n == 1:
                        emit_run_fn(output, word_buf[0], 1)
                    else:
                        emit_literal_fn(output, word_buf[:lit_n])
                    state = _State.CONTEXT_SKIP
                    continue
                emit_run_fn(output, word_buf[0], 1)
                state = _State.CONTEXT_SKIP
                continue

            if word_buf[-1] == w:
                if run_len > 2:
                    emit_literal_fn(output, word_buf[:-1])
                elif run_len == 2:
                    emit_run_fn(output, word_buf[0], 1)
                cur = w
                word_buf = [w]
                run_len = 2
                match_count = min(match_count, 2)
                state = _State.RUN
                continue

            if run_len + 1 > lit_overflow:
                lit_n = run_len - 1
                if lit_n == 1:
                    emit_run_fn(output, word_buf[0], 1)
                else:
                    emit_literal_fn(output, word_buf[:lit_n])
                emit_run_fn(output, word_buf[-1], 1)
                cur = w
                word_buf = [word_buf[-1], w]
                run_len = 2
                match_count = min(match_count, 2)
                state = _State.LITERAL
                continue

            word_buf.append(w)
            run_len += 1
            continue

        if state == _State.CONTEXT_SKIP:
            ref = ctx.shift_and_get_ref()
            w = read_word()
            if w == eof:
                emit_context_skip(output, match_count)
                state = _State.DONE
                continue
            ctx.store(w)
            if ref != w:
                emit_context_skip(output, match_count)
                match_count = 0
                cur = w
                state = _State.MAIN
                continue
            match_count += 1
            continue

        if state == _State.FINALIZE_LIT:
            if match_count < run_len:
                lit_n = run_len - match_count
                if lit_n == 1:
                    emit_run_fn(output, word_buf[0], 1)
                else:
                    emit_literal_fn(output, word_buf[:lit_n])
            else:
                emit_run_fn(output, word_buf[0], 1)
            if match_count > 0:
                emit_context_skip(output, match_count)
            state = _State.DONE
            continue

        if state == _State.DONE:
            break

    return bytes(output)


def finalize_compressed(output: bytes, data: bytes, input_len: int, word_bits: int) -> bytes:
    """Fall back to a raw dump if the compressed output exceeds raw + 0x14 bytes.

    Returns:
        Either `output` unchanged or a raw-dump block carrying `data`.
    """
    if len(output) > input_len + 0x14:
        n_words = (input_len * 8 + word_bits - 1) // word_bits
        padded_bytes = (n_words * word_bits + 7) // 8
        pad_count = padded_bytes - input_len

        out = bytearray()
        if n_words < 0x41:
            out.append((n_words - 2) | 0x40)
        else:
            out.append(0x7F)
            emit_count_ext(out, n_words - 0x41)
        out.extend(data)
        out.extend(bytes(pad_count))
        return bytes(out)
    return bytes(output) if isinstance(output, bytearray) else output
