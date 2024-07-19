import numpy as np
from setuptools import setup, Extension
import sys

setup(
  ext_modules=[
        Extension(
            name="prpy.numpy.image_ops",
            sources=["prpy/numpy/image_ops.c"],
            include_dirs=[np.get_include()]
        ),
    ]
)

# Enforce Python version constraints
python_version = sys.version_info
if any(arg in sys.argv for arg in ['numpy', 'tensorflow', 'torch']):
    if python_version >= (3, 12):
        sys.stderr.write("This package does not support Python 3.12 or higher for the selected extras.\n")
        sys.exit(1)
