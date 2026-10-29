"""Build script for the Cython acceleration modules in `brother_xl2hb._native`.

Build in place with
``uv run --with cython==3.3.0,setuptools==84.2.1 python setup_cython.py build_ext --inplace``
(the versions `cups/install.sh` and CI build with).
"""

from Cython.Build import cythonize
from setuptools import Extension, setup

_PACKAGE = "brother_xl2hb._native"
_MODULES = ("_rle_fast", "_dither_fast", "_band_fast", "_color_fast", "_fine_fast")

setup(
    name="brother-hl4150cdn-cython",
    ext_modules=cythonize(
        [
            Extension(
                f"{_PACKAGE}.{name}",
                sources=[f"src/brother_xl2hb/_native/{name}.pyx"],
                extra_compile_args=["-O3", "-fno-strict-aliasing"],
            )
            for name in _MODULES
        ],
        compiler_directives={"language_level": "3"},
        include_path=["src"],
    ),
    package_dir={"": "src"},
    zip_safe=False,
)
