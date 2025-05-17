# Copyright (c) 2025 Philipp Rouast
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from typing import Union

def div0(
    a: Union[np.ndarray, float, int],
    b: Union[np.ndarray, float, int],
    fill: Union[float, int] = np.nan
  ) -> np.ndarray:
  """
  Divide after accounting for zeros in divisor, e.g.:
  
  - div0( [-1, 0, 1], 0, fill=np.nan) -> [nan nan nan]
  - div0( 1, 0, fill=np.inf ) -> inf
  
  Source: https://stackoverflow.com/a/35696047/3595278
  
  Args:
    a: Dividend
    b: Divisor
    fill: Use this value to fill where b == 0.
  Returns:
    c: safe a/b
  """
  assert isinstance(fill, (int, float))
  with np.errstate(divide='ignore', invalid='ignore'):
    c = np.true_divide(a, b)
  if np.isscalar(c):
    return c if np.isfinite(c) else fill
  else:
    c[~np.isfinite(c)] = fill
    return c
  
def standardize(
    x: np.ndarray,
    axis: Union[int, None] = -1
  ) -> np.ndarray:
  """
  Perform standardization
  
  - Note: Returns zero if std == 0
  
  Args:
    x: The input data
    axis: Axis over which to standardize
  Returns:
    x: The standardized data
  """
  assert axis is None or isinstance(axis, int) or (isinstance(axis, tuple) and all(isinstance(i, int) for i in axis))
  x = np.array(x)
  x -= np.mean(x, axis=axis, keepdims=x.ndim>0)
  std = np.std(x, axis=axis, keepdims=x.ndim>0)
  x = div0(x, std, fill=0)
  return x

def normalize(
    x: np.ndarray,
    axis: Union[int, tuple, None] = -1
  ) -> np.ndarray:
  """
  Perform normalization

  Args:
    x: The input data
    axis: Axis or axes over which to normalize
  Returns:
    x: The normalized data
  """
  assert axis is None or isinstance(axis, int) or (isinstance(axis, tuple) and all(isinstance(i, int) for i in axis))
  x = np.array(x)
  x -= np.mean(x, axis=axis, keepdims=x.ndim>0)
  return x
