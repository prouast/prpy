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
from typing import Callable, Optional

from prpy.numpy.stride_tricks import window_view, resolve_1d_window_view

def rolling_calc(
    x: np.ndarray,
    calc_fn: Callable,
    min_window_size: int,
    max_window_size: int,
    overlap: Optional[int] = None,
    transform_fn: Optional[Callable] = None,
    fill_method: str = 'pad_val',
    pad_val: float = np.nan,
    rolling_pad_mode: str = 'constant'
  ) -> np.ndarray:
  """
  Apply `calc_fn` to `x` using rolling window along its first dim.

  Args:
    x: The values to be processed. Shape (n, ...) -> m dims in total
    calc_fn: The function which should be applied to a window of values.
      - Must accept (m+1)-dim array and reduce all but the first dim.
    min_window_size: The minimum size of the calculation window in number of values.
    max_window_size: The maximum size of the calculation window in number of values.
    overlap: The overlap of the rolling windows (default: max_window_size-1)
    transform_fn: Optional function that is applied before `calc_fn` to the rolling view.
      - Must preserve the view's shape.
    fill_method: Method for filling/padding the result.
      - `pad_val`, `mean`, or `start`
    pad_val: Value to use for filling/padding when fill_method is pad_val
    rolling_pad_mode: Intermediate pad mode to use for end of rolling window view
  Returns:
    The result of `calc_fn` applied to the time diffs. Shape (n,)
  """
  x = np.asarray(x, dtype=float)
  if overlap is None: overlap = max_window_size - 1
  size = x.shape[0]
  if size < min_window_size:
    # Not enough values available -> return pad_val
    result = np.full(size, fill_value=pad_val)
  else:
    # Create a rolling window view on x
    x_view, pad_start, pad_end = window_view(x=x,
                                             min_window_size=min_window_size,
                                             max_window_size=max_window_size,
                                             overlap=overlap,
                                             pad_mode=rolling_pad_mode)
    if transform_fn is not None:
      # Transform the view
      x_view_trans = transform_fn(x_view)
      if x_view_trans.shape != x_view.shape:
        raise ValueError("transform_fn must preserve the view's shape")
      x_view = x_view_trans
    # Apply the calculation
    x_calc = calc_fn(x_view)
    if x_calc.ndim != 1 or x_calc.shape[0] != x_view.shape[0]:
      raise ValueError("calc_fn must reduce all but the first dimension")
    # Resolve the view
    result = resolve_1d_window_view(x=x_calc,
                                    window_size=max_window_size,
                                    overlap=overlap,
                                    pad_start=pad_start,
                                    pad_end=pad_end,
                                    fill_method=fill_method,
                                    pad_val=pad_val)
  assert result.shape[0] == size, f"result.shape[0] {result.shape[0]} != {size} size"
  return result

def rolling_calc_ragged(
    x: np.ndarray,
    calc_fn: Callable[[np.ndarray], float],
    min_window_size: float,
    max_window_size: float,
    *,
    pad_val: float = np.nan
  ) -> np.ndarray:
  """
  Apply an arbitrary reducer `calc_fn` to ragged backward-looking rolling windows of `x`.

  Args:
    x: 1-D increasing ndarray (e.g. detection timestamps)
    calc_fn: Callable that maps ndarray -> scalar
    min_window_size: Minimum size of the backward-looking window (same units as x)
    max_window_size: Maximum size of the backward-looking window (same units as x)
    pad_val: value to emit when window too small
  Returns:
    aligned: ndarray (len(x),) result broadcast / merged onto x
  """
  x = np.asarray(x, dtype=float)
  if x.ndim != 1:
    raise ValueError("x must be 1-D and strictly increasing")
  if min_window_size < 0 or max_window_size < min_window_size:
    raise ValueError("0 ≤ min_window_size ≤ max_window_size required")
  n = x.size
  # left[i] = first index inside (x[i] - max_window_size, x[i]]
  left = np.searchsorted(x, x - max_window_size, side="right")
  # width of each candidate window
  valid = x >= min_window_size
  out = np.full(n, pad_val, dtype=float)
  for i in np.where(valid)[0]:
    l = left[i]
    out[i] = calc_fn(x[l : i + 1])
  return out
