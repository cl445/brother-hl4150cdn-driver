"""Native RLE bit-twiddlers (Rust and Cython): byte-identity vs. the pure-Python fallback."""

import numpy as np
import pytest

from brother_xl2hb.codec.rle import group_bits_py, pack_groups_py
from conftest import native_variants


@pytest.fixture(scope="module", params=native_variants("_rle_fast"))
def native(request):
    """Each built `_rle_fast`: the Rust and the Cython one."""
    return request.param


@pytest.mark.parametrize("group_size", [10, 12, 20])
@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("size", [0, 1, 7, 8, 9, 595, 596, 1024, 8191])
def test_group_bits_identity(native, group_size, seed, size):
    rng = np.random.default_rng(seed)
    data = bytes(rng.integers(0, 256, size, dtype=np.uint8))
    assert native.group_bits(data, group_size) == group_bits_py(data, group_size)


@pytest.mark.parametrize("group_size", [10, 12, 20])
def test_group_bits_edge_patterns(native, group_size):
    for data in [b"", b"\x00", b"\xff", b"\xaa\x55" * 100, b"\x00" * 595, b"\xff" * 595]:
        assert native.group_bits(data, group_size) == group_bits_py(data, group_size), (data[:8], group_size)


@pytest.mark.parametrize("group_size", [10, 12, 20])
@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("n", [0, 1, 2, 17, 397, 477, 596])
def test_pack_groups_identity(native, group_size, seed, n):
    rng = np.random.default_rng(seed)
    max_val = (1 << group_size) - 1
    groups = [int(v) for v in rng.integers(0, max_val + 1, n, dtype=np.uint32)]
    assert native.pack_groups(groups, group_size) == pack_groups_py(groups, group_size)


@pytest.mark.parametrize("group_size", [10, 12, 20])
def test_round_trip(native, group_size):
    rng = np.random.default_rng(123)
    for _ in range(5):
        n_bytes = int(rng.integers(1, 1000))
        data = bytes(rng.integers(0, 256, n_bytes, dtype=np.uint8))
        groups = native.group_bits(data, group_size)
        repacked = native.pack_groups(groups, group_size)
        expected = pack_groups_py(group_bits_py(data, group_size), group_size)
        assert repacked == expected


@pytest.mark.parametrize("group_size", [10, 12, 20])
def test_group_bits_accepts_buffers(native, group_size):
    data = bytes(np.random.default_rng(5).integers(0, 256, 597, dtype=np.uint8))
    expected = group_bits_py(data, group_size)
    for buf in (bytearray(data), memoryview(data)):
        assert native.group_bits(buf, group_size) == expected
        assert group_bits_py(buf, group_size) == expected
    assert native.group_bits(np.frombuffer(data, np.uint8), group_size) == expected
    assert native.group_bits(bytearray(), group_size) == []


@pytest.mark.parametrize("group_size", [1, 31, 32])
def test_group_bits_widest_and_narrowest(native, group_size):
    data = bytes(np.random.default_rng(6).integers(0, 256, 37, dtype=np.uint8))
    groups = native.group_bits(data, group_size)
    assert groups == group_bits_py(data, group_size)
    assert native.pack_groups(groups, group_size) == pack_groups_py(groups, group_size)


@pytest.mark.parametrize("group_size", [-1, 0, 33, 64])
def test_rejects_unsupported_group_size(native, group_size):
    for fn in (native.group_bits, group_bits_py):
        with pytest.raises(ValueError, match="group size"):
            fn(b"\x01\x02\x03\x04\x05", group_size)
    for fn in (native.pack_groups, pack_groups_py):
        with pytest.raises(ValueError, match="group size"):
            fn([1, 2], group_size)
