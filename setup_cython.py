"""Build script for the Cython acceleration modules.

Build in place with
``uv run --with cython,setuptools python setup_cython.py build_ext --inplace``.
"""

from Cython.Build import cythonize
from setuptools import Extension, setup

_MODULES = ("_rle_fast", "_dither_fast", "_band_fast", "_color_fast")

setup(
    name="brother-hl4150cdn-cython",
    ext_modules=cythonize(
        [
            Extension(name, sources=[f"src/{name}.pyx"], extra_compile_args=["-O3", "-fno-strict-aliasing"])
            for name in _MODULES
        ],
        compiler_directives={"language_level": "3"},
        include_path=["src"],
    ),
    package_dir={"": "src"},
    zip_safe=False,
)
