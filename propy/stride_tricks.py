###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, September 2021              #
###############################################################################

import numpy as np

def window_view(x, min_window_size, max_window_size, overlap, pad_mode='constant', const_val=np.nan):
  """Create a window view along its first dim into an n-d array x.
  Args:
    x: The n-d array into which we want to create a windowed view
    min_window_size: The minimum window size
    max_window_size: The maximum window size
    overlap: The overlap of the sliding windows
    pad_mode: The pad mode
    const_val: The constant value to be padded with if pad_mode == 'constant'
  Returns:
    y: The n+1-d windowed view into x
    pad_start: How much padding was applied at the start (scalar)
    pad_end: How much padding was applied at the end (scalar)
  """
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

def reduce_2d_window_view(x, overlap, hanning=False):
  """Reduce a window view by arranging the first dimension as sliding
    windows and then reducing it using the mean.
  Args:
    x: The 2-d window view [n_windows, window_size]
    window_size: The window size with which the window view was created
    overlap: The overlap with which the window view was created
    hanning: Whether to reduce the window view with hanning windows
  Returns:
    mean: The 1-d reduced window view
  """
  # Infer number of windows and original length
  num_windows, window_size = x.shape
  original_len = num_windows * window_size - (num_windows-1) * overlap
  # Apply hanning window to taper x
  if hanning:
    x *= np.hanning(window_size)
  # Add padding to extend the matrix to the dimensions of the diagonal matrix
  y = np.pad(x, ((0, 0), (0, original_len - window_size)), 'constant')
  # Use as_strided to create a view of the diagonal matrix
  #  that aligns the windows temporally as they were generated
  # https://stackoverflow.com/a/60460462/3595278
  y_roll = y[:, [*range(y.shape[1]),*range(y.shape[1]-1)]].copy() #need `copy`
  strd_0, strd_1 = y_roll.strides
  view = np.lib.stride_tricks.as_strided(y_roll, (*y.shape, original_len), (strd_0, strd_1, strd_1))
  step_size = window_size - overlap
  m = np.asarray([step_size * i for i in range(num_windows)])
  view = view[np.arange(y.shape[0]), (original_len-m)%original_len]
  # Merge the windows by taking the mean across the time dimension
  mean = np.true_divide(
    view.sum(0), np.maximum((view != 0).sum(0), 1))
  return mean

def resolve_1d_window_view(x, window_size, overlap, pad_end, fill_method):
  """Resolve an 1-d window view by extending it to the expected shape. This
    is useful if processing on each window created a scalar value.
  Args:
    x: The 1-d window view to be resolved
    window_size: The window size used to create the view
    overlap: The overlap used to create the view
    pad_end: How much padding was applied to the end
    fill_method: Method for filling/padding the result
  Returns:
    vals: The 1-d resolved data
  """
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
    vals = np.concatenate([np.repeat(fill, window_size-1), x])
  elif overlap < window_size - 1:
    # For any other overlaps, we will have to build an intermediate 2-d
    # window view of the vals and reduce it via reduce_2d_window_view.
    n = len(x)
    x = np.reshape(np.repeat(x, window_size), (n, window_size))
    vals = reduce_2d_window_view(x, overlap=overlap)
    if pad_end > 0:
      # Trim end if it has been padded
      vals = vals[:-pad_end]
  return vals
