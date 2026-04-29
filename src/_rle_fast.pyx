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
