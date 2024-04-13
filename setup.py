from setuptools import setup, Extension
import numpy as np

setup(
  ext_modules=[
        Extension(
            name="prpy.numpy.image_ops",
            sources=["prpy/numpy/image_ops.c"],
            include_dirs=[np.get_include()]
        ),
    ]
)
