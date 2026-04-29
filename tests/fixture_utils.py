"""Utility for reading zstd-compressed test fixtures.

Fixtures live in ``tests/fixtures/`` as ``.zst`` files.
Callers should ``pytest.skip()`` when ``read_fixture()`` returns ``None``.
"""

from pathlib import Path

import zstandard

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def read_fixture(name: str) -> bytes | None:
    """Read a zstd-compressed fixture by its original filename.

    Args:
        name: Original filename (e.g. ``"a4_white.xl2hb"``).

    Returns:
        Decompressed bytes, or ``None`` if the fixture file is missing.
    """
    path = FIXTURES_DIR / f"{name}.zst"
    if not path.exists():
        return None
    return zstandard.decompress(path.read_bytes())
