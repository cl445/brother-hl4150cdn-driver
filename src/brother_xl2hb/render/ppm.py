"""PPM (P6 binary) reader.

Supports concatenated PPM streams — Ghostscript writes one PPM per page
into the same stdout, so a CUPS pipeline can iterate page-by-page.

The header follows the netpbm format: the magic number, width, height
and maxval separated by any whitespace, where a "#" starts a comment up
to the end of the line, and exactly one whitespace byte before the
raster. Whitespace between concatenated images is skipped.
"""

from typing import BinaryIO

_WHITESPACE = frozenset(b" \t\n\v\f\r")
_HASH = ord("#")
# Longer than any magic number or 32-bit field: stops a stream that is no
# PPM at all from being read as one huge token.
_MAX_TOKEN = 16


def _read_byte(stream: BinaryIO) -> int:
    """Read one header byte.

    Returns:
        The byte's value.

    Raises:
        ValueError: At EOF, since the header is not complete yet.
    """
    byte = stream.read(1)
    if not byte:
        msg = "PPM header incomplete"
        raise ValueError(msg)
    return byte[0]


def _skip_comment(stream: BinaryIO) -> None:
    """Skip a comment up to and including its line end."""
    while _read_byte(stream) not in b"\n\r":
        pass


def _read_token(stream: BinaryIO) -> bytes:
    """Read the next header token and the one byte that ends it.

    Leading whitespace and comments are skipped. A comment right after
    the token ends it as well, together with its line end.

    Returns:
        The token, or b"" if the stream ends before a token starts.

    Raises:
        ValueError: If the stream ends inside the token or a comment, or
            the token is longer than any header field can be.
    """
    byte = stream.read(1)
    while byte and (byte[0] in _WHITESPACE or byte[0] == _HASH):
        if byte[0] == _HASH:
            _skip_comment(stream)
        byte = stream.read(1)
    if not byte:
        return b""
    token = bytearray(byte)
    while (c := _read_byte(stream)) not in _WHITESPACE:
        if c == _HASH:
            _skip_comment(stream)
            break
        token.append(c)
        if len(token) > _MAX_TOKEN:
            msg = f"PPM header: token too long: {bytes(token)!r}"
            raise ValueError(msg)
    return bytes(token)


def _read_number(stream: BinaryIO, name: str) -> int:
    """Read a positive decimal header field.

    Returns:
        The field's value.

    Raises:
        ValueError: If the header ends or the field is not a positive number.
    """
    token = _read_token(stream)
    if not token:
        msg = "PPM header incomplete"
        raise ValueError(msg)
    if not token.isdigit() or int(token) == 0:
        msg = f"PPM header: invalid {name} {token!r}"
        raise ValueError(msg)
    return int(token)


def read_ppm_header(stream: BinaryIO) -> tuple[int, int, int] | None:
    """Read a P6 PPM header, leaving `stream` at the first pixel byte.

    Returns:
        Tuple `(width, height, maxval)`, or None at EOF.

    Raises:
        ValueError: If the magic bytes are not "P6", the header is
            malformed or incomplete, or maxval is not 255.
    """
    magic = _read_token(stream)
    if not magic:
        return None
    if magic != b"P6":
        msg = f"Not PPM P6 format: {magic!r}"
        raise ValueError(msg)

    width = _read_number(stream, "width")
    height = _read_number(stream, "height")
    maxval = _read_number(stream, "maxval")
    if maxval != 255:
        msg = f"Only 8-bit PPM (maxval=255) supported, got maxval={maxval}"
        raise ValueError(msg)
    return width, height, maxval


def read_ppm(stream: BinaryIO) -> tuple[int, int, int, bytes] | None:
    """Read a PPM file (P6, binary) from a stream.

    Returns:
        Tuple `(width, height, maxval, pixel_data)`, or None at EOF.

    Raises:
        ValueError: If the magic bytes are not "P6", the header is
            malformed, maxval is not 255, or the payload is shorter than
            the header indicates.
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
