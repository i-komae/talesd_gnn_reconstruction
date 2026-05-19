from __future__ import annotations

import sys

import numpy
import pybind11
from setuptools import Extension, setup


compile_args = ["-std=c++17", "-O3"]
if sys.platform == "darwin":
    compile_args.extend(["-mmacosx-version-min=11.0"])


setup(
    ext_modules=[
        Extension(
            "talesd_gnn_reconstruction._collate_ext",
            ["src/talesd_gnn_reconstruction/cpp/collate_ext.cpp"],
            include_dirs=[pybind11.get_include(), numpy.get_include()],
            define_macros=[("TALESD_GNN_COLLATE_MODULE", "_collate_ext")],
            language="c++",
            extra_compile_args=compile_args,
        )
    ],
)
