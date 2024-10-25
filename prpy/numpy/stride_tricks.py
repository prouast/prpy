# Copyright (c) 2024 Philipp Rouast
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
from typing import Union, Tuple

def window_view(
    x: np.ndarray,
    min_window_size: int,
    max_window_size: int,
    overlap: int,
    pad_mode: str = 'constant',
    const_val: Union[float, int] = np.nan
  ) -> Tuple[np.ndarray, int, int]:
  """Create a window view into an n-d array `x` along its first dim.

  Args:
    x: The n-d array into which we want to create a windowed view
    min_window_size: The minimum window size
    max_window_size: The maximum window size
    overlap: The overlap of the sliding windows
    pad_mode: The pad mode
    const_val: The constant value to be padded with if pad_mode == 'constant'
  Returns:
    Tuple of
     - y: The n+1-d windowed view into x of shape (n_windows, window_size, ...)
     - pad_start: How much padding was applied at the start (scalar)
     - pad_end: How much padding was applied at the end (scalar)
  """
  assert isinstance(min_window_size, int) and min_window_size >= 0
  assert isinstance(max_window_size, int) and max_window_size >= min_window_size
  assert isinstance(overlap, int) and overlap < max_window_size, "overlap must be smaller than max_window_size"
  assert isinstance(pad_mode, str)
  assert isinstance(const_val, (float, int))
  x = np.asarray(x)
  original_len = x.shape[0]
  step_size = max_window_size - overlap
  # Pad the start using variable window sizes
  pad_start = max_window_size - min_window_size
  # Pad the end if there is a remainder
  remainder = (original_len - min_window_size) % step_size
  pad_end = 0 if remainder == 0 else step_size - remainder
  pad_width = ((pad_start, pad_end),) + ((0, 0),) * (x.ndim - 1)
  if pad_mode=='constant':
    x = np.pad(x, pad_width, mode=pad_mode, constant_values=(const_val,))
  else:
    x = np.pad(x, pad_width, mode=pad_mode)
  # Calculate the shape of the view for the first dimension
  new_shape = ((pad_start + original_len + pad_end - max_window_size) // step_size + 1, max_window_size)
  # Add shapes for following dimensions
  new_shape += x.shape[1:]
  # Calculate the stride of the view for the first dimension
  new_strides = ((step_size * x.strides[0],) + (x.strides[0],))
  # Add strides for following dimensions
  new_strides += x.strides[1:]
  # Generate the view
  y = np.lib.stride_tricks.as_strided(
    x, shape=new_shape, strides=new_strides)
  # Return
  return y, pad_start, pad_end

def reduce_window_view(
    x: np.ndarray,
    overlap: int,
    pad_end: int = 0,
    hanning: bool = False
  ) -> np.ndarray:
  """Reduce an n-d window view by arranging the first dimension as sliding
  windows and then reducing it using the mean.

  Args:
    x: The n-d window view of shape (n_windows, window_size, ...)
    overlap: The overlap with which the window view was created
    pad_end: How much padding was applied to the end when the window view was created
    hanning: Whether to reduce the window view with hanning windows
  Returns:
    mean: The n-1-d reduced window view [original_len, ...]
  """
  assert isinstance(x, np.ndarray)
  assert isinstance(hanning, bool)
  # Infer number of windows and original length (including extra padding)
  num_windows = x.shape[0]
  window_size = x.shape[1]
  assert isinstance(overlap, int) and overlap >= 0 and overlap < window_size
  assert isinstance(pad_end, int) and pad_end >= 0 and pad_end < window_size
  original_len_with_pad_end = num_windows * window_size - (num_windows-1) * overlap
  # Apply hanning window to taper x
  if hanning:
    x *= np.hanning(window_size)
  # Add padding to extend the matrix to the dimensions of the diagonal matrix
  padding = ((0, 0), (0, original_len_with_pad_end - window_size)) + ((0, 0),) * (x.ndim - 2)
  y = np.pad(x, padding, 'constant')
  # Use as_strided to create a view of the diagonal matrix
  #  that aligns the windows temporally as they were generated
  # https://stackoverflow.com/a/60460462/3595278
  y_roll = y[:, [*range(y.shape[1]),*range(y.shape[1]-1)]].copy() #need `copy`
  view_strides = list(y_roll.strides)
  view_strides.insert(1, y_roll.strides[1])
  view_strides = tuple(view_strides)
  view_shape = list(y.shape)
  view_shape.insert(1, original_len_with_pad_end)
  view_shape = tuple(view_shape)
  view = np.lib.stride_tricks.as_strided(y_roll, view_shape, view_strides)
  step_size = window_size - overlap
  m = np.asarray([step_size * i for i in range(num_windows)])
  view = view[np.arange(y.shape[0]), (original_len_with_pad_end-m)%original_len_with_pad_end]
  # Merge the windows by taking the mean across the time dimension
  mean = np.true_divide(
    view.sum(0), np.maximum((view != 0).sum(0), 1))
  # Trim result by pad_end if necessary
  if pad_end > 0: mean = mean[:-pad_end]
  return mean

def resolve_1d_window_view(
    x: np.ndarray,
    window_size: int,
    overlap: int,
    pad_end: int,
    fill_method: str
  ) -> np.ndarray:
  """Resolve an 1-d window view by extending it to the expected shape.
  
  - This is useful if processing on each window created a scalar value.

  Args:
    x: The 1-d window view to be resolved
    window_size: The window size used to create the view
    overlap: The overlap used to create the view
    pad_end: How much padding was applied to the end
    fill_method: Method for filling/padding the result
  Returns:
    vals: The 1-d resolved data
  """
  assert isinstance(x, np.ndarray) and len(x.shape) == 1
  assert isinstance(window_size, int) and window_size > 0
  assert isinstance(overlap, int) and overlap >= 0 and overlap < window_size
  assert isinstance(pad_end, int) and pad_end >= 0 and pad_end < window_size
  assert isinstance(fill_method, str)
  if overlap == 0:
    # If overlap is zero, we simply need to repeat each value to match window_size
    vals = np.repeat(x, window_size)
    if pad_end > 0:
      # Trim end if it has been padded
      vals = vals[:-pad_end]
  elif overlap == window_size - 1:
    # If overlap is one less than the window size, then values are mostly
    # already ok. We just have to fill the start.
    if fill_method == 'zero':
      fill = 0.0
    elif fill_method == 'mean':
      fill = np.mean(x)
    elif fill_method == 'start':
      fill = x[0]
    else:
      raise ValueError("fill_method {} not supported".format(fill_method))
    vals = np.concatenate([np.repeat(fill, window_size-1), x])
  elif overlap < window_size - 1:
    # For any other overlaps, we will have to build an intermediate 2-d
    # window view of the vals and reduce it via reduce_window_view.
    n = len(x)
    x = np.reshape(np.repeat(x, window_size), (n, window_size))
    vals = reduce_window_view(x, overlap=overlap)
    if pad_end > 0:
      # Trim end if it has been padded
      vals = vals[:-pad_end]
  return vals
