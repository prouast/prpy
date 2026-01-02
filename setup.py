import numpy as np
from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            name="prpy.numpy.image_ops",
            sources=["prpy/numpy/image_ops.c"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_TARGET_VERSION", "NPY_1_24_API_VERSION")],
        ),
    ]
)
