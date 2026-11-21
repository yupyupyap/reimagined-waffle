from __future__ import annotations

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ext_modules = [
    Pybind11Extension(
        "localchessengine_trainer._binpack_native",
        ["src/localchessengine_trainer/native/binpack.cpp"],
        cxx_std=17,
    )
]


setup(cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
