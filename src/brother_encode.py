"""Brother HL-4150CDN XL2HB raster encoder.

Plane-specific RLE codecs (each scanline is encoded into its own block):
  C: 20-bit sliding-window RLE, 3-word context, header QUANT type=2 comp=20
  M: 10-bit sliding-window RLE, 5-word context, header QUANT type=4 comp=10
  Y: 12-bit sliding-window RLE, 3-word context, header QUANT type=2 comp=12
  K: 12-bit sliding-window RLE, 3-word context, header QUANT type=2 comp=12

This module is a thin façade that re-exports the public encoder API
from :mod:`rle`, :mod:`plane_encoders`, and :mod:`fine_encoder`.
"""

from fine_encoder import compress_jpegls_encode, compress_rle_preencode, encode_fine_plane
from plane_encoders import encode_c_plane, encode_m_plane_10, encode_m_plane_20, encode_plane
from rle import group_bits, pack_groups
from xl2hb import BPL

__all__ = [
    "BPL",
    "compress_jpegls_encode",
    "compress_rle_preencode",
    "encode_c_plane",
    "encode_fine_plane",
    "encode_m_plane_10",
    "encode_m_plane_20",
    "encode_plane",
    "group_bits",
    "pack_groups",
]
