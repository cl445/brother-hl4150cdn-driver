"""Brother XL2HB printer driver: PPM pages in, PJL + XL2HB byte stream out.

Subpackages: `codec` (plane RLE and Fine codecs), `color` (colour grids,
saturation, dither), `protocol` (XL2HB stream and PJL), `render` (pages,
bands, the job pipeline) and `_native` (optional Cython kernels with
pure-Python fallbacks). `blobs` locates the tables extracted from Brother's
driver package.
"""
