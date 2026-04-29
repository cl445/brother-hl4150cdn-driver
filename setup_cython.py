"""Build script for the Cython acceleration module.

    uv run --with cython,setuptools python setup_cython.py build_ext --inplace
"""

from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    name="brother-hl4150cdn-cython",
    ext_modules=cythonize(
        [
            Extension(
                "_rle_fast",
                sources=["src/_rle_fast.pyx"],
                extra_compile_args=["-O3", "-fno-strict-aliasing"],
            ),
        ],
        compiler_directives={"language_level": "3"},
    ),
    package_dir={"": "src"},
    zip_safe=False,
)
