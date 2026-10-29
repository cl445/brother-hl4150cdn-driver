"""Brother XL2HB printer driver: PPM pages in, PJL + XL2HB byte stream out.

Subpackages: `codec` (plane RLE and Fine codecs), `color` (colour grids,
saturation, dither), `protocol` (XL2HB stream and PJL), `models` (what each
printer's filter writes for paper and media), `render` (one page: rows, the
per-line and the band path) and `_native` (optional Cython kernels with
pure-Python fallbacks). `job` runs a whole print job; `blobs` locates the
tables extracted from Brother's driver package.
"""
