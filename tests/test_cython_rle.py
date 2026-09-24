"""Cython RLE bit-twiddlers: byte-identity vs. the pure-Python fallback."""

import importlib

import numpy as np
import pytest

import rle as rle_module
from rle import group_bits_py, pack_groups_py

pytestmark = pytest.mark.skipif(
    not rle_module.HAS_CYTHON_RLE,
    reason="Cython _rle_fast not built — run `python setup_cython.py build_ext --inplace`",
)


@pytest.fixture(scope="module")
def cy():
    return importlib.import_module("_rle_fast")


@pytest.mark.parametrize("group_size", [10, 12, 20])
@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("size", [0, 1, 7, 8, 9, 595, 596, 1024, 8191])
def test_group_bits_identity(cy, group_size, seed, size):
    rng = np.random.default_rng(seed)
    data = bytes(rng.integers(0, 256, size, dtype=np.uint8))
    assert cy.group_bits(data, group_size) == group_bits_py(data, group_size)


@pytest.mark.parametrize("group_size", [10, 12, 20])
def test_group_bits_edge_patterns(cy, group_size):
    for data in [b"", b"\x00", b"\xff", b"\xaa\x55" * 100, b"\x00" * 595, b"\xff" * 595]:
        assert cy.group_bits(data, group_size) == group_bits_py(data, group_size), (data[:8], group_size)


@pytest.mark.parametrize("group_size", [10, 12, 20])
@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("n", [0, 1, 2, 17, 397, 477, 596])
def test_pack_groups_identity(cy, group_size, seed, n):
    rng = np.random.default_rng(seed)
    max_val = (1 << group_size) - 1
    groups = [int(v) for v in rng.integers(0, max_val + 1, n, dtype=np.uint32)]
    assert cy.pack_groups(groups, group_size) == pack_groups_py(groups, group_size)


@pytest.mark.parametrize("group_size", [10, 12, 20])
def test_round_trip(cy, group_size):
    rng = np.random.default_rng(123)
    for _ in range(5):
        n_bytes = int(rng.integers(1, 1000))
        data = bytes(rng.integers(0, 256, n_bytes, dtype=np.uint8))
        groups = cy.group_bits(data, group_size)
        repacked = cy.pack_groups(groups, group_size)
        expected = pack_groups_py(group_bits_py(data, group_size), group_size)
        assert repacked == expected
