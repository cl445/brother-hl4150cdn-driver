"""PPM (P6 binary) reader.

Supports concatenated PPM streams — Ghostscript writes one PPM per page
into the same stdout, so a CUPS pipeline can iterate page-by-page.
"""

from typing import BinaryIO


def read_ppm_header(stream: BinaryIO) -> tuple[int, int, int] | None:
    """Read a P6 PPM header, leaving `stream` at the first pixel byte.

    Returns:
        Tuple `(width, height, maxval)`, or None at EOF.

    Raises:
        ValueError: If the magic bytes are not "P6" or maxval is not 255.
    """
    magic = stream.readline()
    if not magic or not magic.strip():
        return None
    magic = magic.strip()
    if magic != b"P6":
        msg = f"Not PPM P6 format: {magic!r}"
        raise ValueError(msg)

    while True:
        line = stream.readline().strip()
        if not line.startswith(b"#"):
            break
    parts = line.split()
    if len(parts) == 2:
        width, height = int(parts[0]), int(parts[1])
    else:
        width = int(parts[0])
        height = int(stream.readline().strip())

    maxval = int(stream.readline().strip())
    if maxval != 255:
        msg = f"Only 8-bit PPM (maxval=255) supported, got maxval={maxval}"
        raise ValueError(msg)
    return width, height, maxval


def read_ppm(stream: BinaryIO) -> tuple[int, int, int, bytes] | None:
    """Read a PPM file (P6, binary) from a stream.

    Returns:
        Tuple `(width, height, maxval, pixel_data)`, or None at EOF.

    Raises:
        ValueError: If the magic bytes are not "P6", maxval is not 255,
            or the payload is shorter than the header indicates.
    """
    header = read_ppm_header(stream)
    if header is None:
        return None
    width, height, maxval = header

    expected = width * height * 3
    data = stream.read(expected)
    if len(data) != expected:
        msg = f"PPM payload incomplete: got {len(data)} bytes, expected {expected}"
        raise ValueError(msg)
    return width, height, maxval, data
