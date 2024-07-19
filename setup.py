import numpy as np
from setuptools import setup, Extension

setup(
  ext_modules=[
        Extension(
            name="prpy.numpy.image_ops",
            sources=["prpy/numpy/image_ops.c"],
            include_dirs=[np.get_include()]
        ),
    ]
)
