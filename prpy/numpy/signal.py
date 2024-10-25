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

import logging
import math
import numpy as np
from scipy import signal, interpolate, fft, stats
from scipy.sparse import spdiags
from scipy.ndimage import uniform_filter1d
from typing import Union, Tuple, Callable

from prpy.numpy.stride_tricks import window_view, resolve_1d_window_view

def div0(
    a: Union[np.ndarray, float, int],
    b: Union[np.ndarray, float, int],
    fill: Union[float, int] = np.nan
  ) -> np.ndarray:
  """Divide after accounting for zeros in divisor, e.g.:
  
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

def normalize(
    x: np.ndarray,
    axis: Union[int, tuple, None] = -1
  ) -> np.ndarray:
  """Perform normalization

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

def standardize(
    x: np.ndarray,
    axis: Union[int, None] = -1
  ) -> np.ndarray:
  """Perform standardization
  
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

def moving_average(
    x: np.ndarray,
    size: int,
    axis: Union[int, None] = -1,
    pad_method: str = 'reflect',
    scale: bool = False,
    scale_factor: float = 1000000000.
  ) -> np.ndarray:
  """Perform moving average

  Args:
    x: The input data
    size: The size of the moving average window
    axis: Axis over which to calculate moving average
    pad_method: Method for padding ends to keep same dims
    scale: Internally scale the input data up before applying filter
    scale_factor: Factor to use for scaling
  Returns:
    x: The averaged data
  """
  assert axis is None or isinstance(axis, int) or (isinstance(axis, tuple) and all(isinstance(i, int) for i in axis))
  assert isinstance(size, int) and size > 0
  assert isinstance(scale, bool)
  assert isinstance(scale_factor, float)
  x = np.array(x)
  if np.isnan(x).any():
    return x
  if scale:
    x *= scale_factor
  y = uniform_filter1d(x, size, mode=pad_method, origin=0, axis=axis)
  if scale:
    y /= scale_factor
  return y

def moving_average_size_for_response(
    sampling_freq: Union[float, int],
    cutoff_freq: Union[float, int]
  ) -> int:
  """Estimate the required moving average size to achieve a given response

  Args:
    sampling_freq: The sampling frequency [Hz]
    cutoff_freq: The desired cutoff frequency [Hz]
  Returns:
    size: The estimated moving average size
  """
  assert isinstance(sampling_freq, (float, int))
  assert isinstance(cutoff_freq, (float, int))
  assert cutoff_freq > 0, "Cutoff frequency needs to be greater than zero"
  # Adapted from https://dsp.stackexchange.com/a/14648
  # cutoff freq in Hz
  F = cutoff_freq / sampling_freq
  size = int(math.sqrt(0.196202 + F * F) / F)
  return max(size, 1)

def moving_std(
    x: np.ndarray,
    size: int,
    overlap: int,
    fill_method: str = 'mean'
  ) -> np.ndarray:
  """Compute moving standard deviation

  Args:
    x: The data to be computed. Shape (n,)
    size: The size of the moving window
    overlap: The overlap of the moving windows
    fill_method: Method to fill the edges.
      Options: 'zero', 'mean' (default), or 'start'
  Returns:
    std: The moving standard deviations
  """
  x = np.asarray(x)
  assert len(x.shape) == 1, "Only 1-D arrays supported"
  assert isinstance(size, int)
  assert isinstance(overlap, int)
  x_view, _, pad_end = window_view(
    x=x,
    min_window_size=size,
    max_window_size=size,
    overlap=overlap)
  y_view = np.std(x_view, axis=-1)
  y = resolve_1d_window_view(
    x=y_view,
    window_size=size,
    overlap=overlap,
    pad_end=pad_end,
    fill_method=fill_method)
  return y

def detrend(
    z: np.ndarray,
    Lambda: int,
    axis: int = -1
  ) -> np.ndarray:
  """Detrend signal(s)
  
  Vectorized implementation of the detrending method by
    Tarvainen et al. (2002). Based on code listing in the Appendix.

  Args:
    z: The input signal. Shape (b, n) or (n,)
    Lambda: The lambda parameter
    axis: The axis along which should be detrended
  Returns:
    proc_z: The detrended signal
  """
  assert isinstance(Lambda, int)
  assert isinstance(axis, int) and (axis == 0 or axis == 1 or axis == -1)
  axis = 1 if axis == -1 else axis
  z = np.asarray(z) # Make sure z is np array
  z = np.nan_to_num(z) # Replace NAs with 0
  if len(z.shape) == 1:
    z = np.expand_dims(z, axis=1-axis)
  assert z.ndim == 2, "z.ndim must equal 2"
  T = z.shape[axis]
  if T < 3:
   return z
  # Identity matrix
  I = np.identity(T)
  # Regularization matrix
  D2 = spdiags(
    [np.ones(T), -2*np.ones(T), np.ones(T)],
    [0, 1, 2], (T-2), T).toarray()
  # Inverse of I+lambda^2*D2’*D2
  inv = np.linalg.inv(I + (Lambda**2) * np.dot(D2.T, D2))
  # Compute the detrending operation (vectorized)
  if axis == 0:
    z = np.transpose(z)
  proc_z = np.matmul((I - inv), z.T)
  if axis == 1:
    proc_z = np.transpose(proc_z)
  # Squeeze if necessary
  proc_z = np.squeeze(proc_z)
  # Return
  return proc_z

# TODO(prouast): Write tests
def windowed_standardize(
    x: np.ndarray,
    window_size: int,
    windowed_mean: bool = True,
    windowed_std: bool = True
  ) -> np.ndarray:
  """Perform dynamic standardization based on windowed mean and std

  Args:
    x: The input data
    window_size: The size of the moving window
    windowed_mean: Boolean indicating whether mean should be windowed
    windowed_std: Boolean indicating whether std should be windowed
  Returns:
    y: The standardized data
  """
  x = np.array(x)
  if windowed_mean:
    mean = moving_average(x, size=window_size, scale=True)
  else:
    mean = np.mean(x)
  if windowed_std:
    std = moving_std(x, size=window_size, overlap=window_size-1)
  else:
    std = np.std(x)
  x -= mean
  x /= std
  return x

# TODO(prouast): Write tests
def butter_bandpass(
    x: np.ndarray,
    lowcut: Union[int, float],
    highcut: Union[int, float],
    fs: Union[int, float],
    axis: Union[int, tuple, None] = -1,
    order: int = 5
  ) -> np.ndarray:
  """Apply a butterworth bandpass filter.

  Args:
    x: The signal data
    lowcut: The lower cutoff frequency
    highcut: The higher cutoff frequency
    fs: The sampling frequency
    axis: The axis along which to apply the filter
    order: The order of the filter
  Returns:
    y: The filtered signal data
  """
  assert axis is None or isinstance(axis, int) or (isinstance(axis, tuple) and all(isinstance(i, int) for i in axis))
  assert isinstance(lowcut, (int, float))
  assert isinstance(highcut, (int, float))
  assert isinstance(fs, (int, float))
  assert isinstance(order, int)
  def butter_bandpass_filter(
      lowcut: Union[int, float],
      highcut: Union[int, float],
      fs: Union[int, float],
      order: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return signal.butter(order, [low, high], btype='band')
  b, a = butter_bandpass_filter(
    lowcut=lowcut, highcut=highcut, fs=fs, order=order)
  y = signal.lfilter(
    b=b, a=a, x=x, axis=axis)
  return y

def estimate_freq(
    x: np.ndarray,
    f_s: Union[float, int],
    f_range: tuple = None,
    f_res: float = None,
    method: str = 'fft',
    max_periodicity_deviation: float = 0.5,
    axis: int = -1
  ) -> np.ndarray:
  """Determine maximum frequencies in x.

  Args:
    x: The signal data. Shape: (n_data,) or (n_sig, n_data)
    f_s: The sampling frequency [Hz]
    f_range: Optional expected range of freqs [Hz] - (min, max)
    f_res: Optional frequency resolution for analysis [Hz]
      (useful if signal small; applies only to periodogram)
    method: The method to be used [fft, peak, or periodogram]
    max_periodicity_deviation: Maximum relative deviation of peaks from regular periodicity
    axis: The axis along which to estimate frequencies
  Returns:
    f_out: The maximum frequencies [Hz]. Shape: (n_sig,)
  """
  assert isinstance(method, str)
  if method == 'fft':
    return estimate_freq_fft(x, f_s=f_s, f_range=f_range, axis=axis)
  elif method == 'peak':
    return estimate_freq_peak(x, f_s=f_s, f_range=f_range, max_periodicity_deviation=max_periodicity_deviation, axis=axis)
  elif method == 'periodogram':
    return estimate_freq_periodogram(x, f_s=f_s, f_range=f_range, f_res=f_res, axis=axis)
  else:
    return ValueError("method should be 'peak', 'fft', or 'periodogram' but was {}".format(method))

def estimate_freq_fft(
    x: np.ndarray,
    f_s: Union[float, int],
    f_range: Union[tuple, None] = None,
    axis: int = -1
  ) -> np.ndarray:
  """Use a fourier transform to determine maximum frequencies.
  
  Args:
    x: The signal data. Shape: (n_data,) or (n_sig, n_data)
    f_s: The sampling frequency [Hz]
    f_range: Optional expected range of freqs [Hz] - (min, max)
    axis: The axis along which to estimate frequencies
  Returns:
    f_out: The maximum frequencies [Hz]. Shape: (n_sig,)
  """
  assert isinstance(f_s, (float, int)) and f_s > 0
  assert f_range is None or (isinstance(f_range, tuple) and len(f_range) == 2 and all(isinstance(i, (int, float)) for i in f_range))
  assert isinstance(axis, int) and (axis == 0 or axis == 1 or axis == -1)
  x = np.asarray(x)
  # Change to 2-dim array if necessary
  if len(x.shape) == 1:
    x = np.expand_dims(x, axis=0)
  # Run the fourier transform
  w = fft.rfft(x, axis=axis)
  f = fft.rfftfreq(x.shape[axis], 1/f_s)
  # Restrict by range if necessary
  if f_range is not None:
    # Bandpass: Set w outside of range to zero
    f_min = min(np.amax(f), f_range[0])
    f_max = max(np.amin(f), f_range[1])
    w = np.where(np.logical_or(f < f_min, f > f_max), 0, w)
  # Determine maximum frequency component
  idx = np.argmax(np.abs(w), axis=axis)
  # Derive frequency in Hz
  f_out = abs(f[idx])
  # Squeeze if necessary
  f_out = np.squeeze(f_out)
  # Return
  return f_out

def estimate_freq_peak(
    x: np.ndarray,
    f_s: Union[float, int],
    f_range: Union[tuple, None] = None,
    max_periodicity_deviation: float = 0.5,
    axis: int = -1
  ) -> np.ndarray:
  """Use peak detection to determine maximum frequencies in x.

  Args:
    x: The signal data. Shape: (n_data,) or (n_sig, n_data)
    f_s: The sampling frequency [Hz]
    f_range: Optional expected range of freqs [Hz] - (min, max)
    max_periodicity_deviation: Maximum relative deviation of peaks from regular periodicity
    axis: The axis along which to estimate frequencies
  Returns:
    f_out: The maximum frequencies [Hz]. Shape: (n_sig,)
  """
  assert isinstance(f_s, (float, int)) and f_s > 0
  assert f_range is None or (isinstance(f_range, tuple) and len(f_range) == 2 and all(isinstance(i, (int, float)) for i in f_range))
  assert isinstance(max_periodicity_deviation, float)
  assert isinstance(axis, int) and (axis == 0 or axis == 1 or axis == -1)
  x = np.asarray(x)
  # Change to 2-dim array if necessary
  if len(x.shape) == 1:
    x = np.expand_dims(x, axis=0)
  # Derive minimum distance between peaks if necessary
  min_dist = max(1/f_range[1]*f_s*(1-max_periodicity_deviation), 0) if f_range is not None else 0
  # Peak detection is only available for 1-D tensors
  def estimate_freq_peak_for_single_axis(x):
    # Find peaks in the signal
    det_idxs, _ = signal.find_peaks(x, height=0, distance=min_dist)
    # Calculate mean distance between peaks
    mean_idx_dist = np.mean(np.diff(det_idxs), axis=-1)
    # Derive the frequency
    return f_s/mean_idx_dist
  # Apply function
  f_out = np.apply_along_axis(estimate_freq_peak_for_single_axis, axis=axis, arr=x)
  # Squeeze if necessary
  f_out = np.squeeze(f_out)
  # Return
  return f_out

def estimate_freq_periodogram(
    x: np.ndarray,
    f_s: Union[float, int],
    f_range: Union[tuple, None] = None,
    f_res: Union[float, None] = None,
    axis: int = -1
  ) -> np.ndarray:
  """Use a periodigram to estimate maximum frequencies at f_res.
  
  - When signal is sampled at a lower frequency than f_res, this is essentially done
    by interpolating in the frequency domain.
  
  Args:
    x: The signal data. Shape: (n_data,) or (n_sig, n_data)
    f_s: The sampling frequency [Hz]
    f_range: Optional expected range of freqs [Hz] - (min, max)
    f_res: Optional frequency resolution for analysis [Hz]
    axis: The axis along which to estimate frequencies
  Returns:
    f_out: The maximum frequencies [Hz]. Shape: (n_sig,)
  """
  assert isinstance(f_s, (float, int)) and f_s > 0
  assert f_range is None or (isinstance(f_range, tuple) and len(f_range) == 2 and all(isinstance(i, (int, float)) for i in f_range))
  assert f_res is None or (isinstance(f_res, (float, int)) and f_res > 0)
  assert isinstance(axis, int) and (axis == 0 or axis == 1 or axis == -1)
  x = np.asarray(x)
  # Change to 2-dim array if necessary
  if len(x.shape) == 1:
    x = np.expand_dims(x, axis=0)
  # Determine the length of the fft if f_res specified
  # Large nfft > x.length leads to zero padding of x before fft (like interpolating frequency domain)
  nfft = None if f_res is None else int(f_s // f_res)
  # Compute
  f, pxx = signal.periodogram(x, fs=f_s, nfft=nfft, detrend=False, axis=axis)
  # Restrict by range if necessary
  if f_range is not None:
    # Bandpass: Set w outside of range to zero
    f_min = min(np.amax(f), f_range[0])
    f_max = max(np.amin(f), f_range[1])
    pxx = np.where(np.logical_or(f < f_min, f > f_max), 0, pxx)
  # Determine maximum frequency component
  idx = np.argmax(pxx, axis=axis)
  # Maximum frequency in Hz
  f_out = f[idx]
  # Squeeze if necessary
  f_out = np.squeeze(f_out)
  # Return
  return f_out

def interpolate_vals(
    x: np.ndarray,
    val_fn: Callable[[np.ndarray], np.ndarray] = lambda x: np.isnan(x)
  ) -> np.ndarray:
  """Linearly interpolate vals matching val_fn

  Args:
    x: The values, shape (n_vals,)
    val_fn: The function, values matching which will be interpolated
  Returns:
    x: The interpolated values, shape (n_vals,)
  """
  assert callable(val_fn)
  x = np.array(x)
  assert len(x.shape) == 1, "Only 1-D arrays supported"
  if val_fn(x).all():
    logging.debug("All elements in x fulfilled val_fn. Not doing anything.")
    return x
  mask = val_fn(x)
  x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
  return x

def interpolate_cubic_spline(
    x: np.ndarray,
    y: np.ndarray,
    xs: np.ndarray,
    axis: int = 0
  ) -> np.ndarray:
  """Interpolate data with a cubic spline.
  Args:
    x: The x values of the data we want to interpolate. 1-dim.
    y: The y values of the data we want to interpolate. Along the given axis,
      shape of y must match shape of x.
    xs: The x values at which we want to interpolate. 1-dim.
  Returns:
    ys: The interpolated y values
  """
  assert isinstance(axis, int)
  x = np.asarray(x)
  y = np.asarray(y)
  xs = np.asarray(xs)
  x = np.nan_to_num(x) # Replace NAs with 0
  y = np.nan_to_num(y) # Replace NAs with 0
  if np.array_equal(x, xs):
    return y
  cs = interpolate.CubicSpline(x, y, axis=axis)
  return cs(xs)

def interpolate_linear_sequence_outliers(
    t: np.ndarray,
    max_diff_rel: float = 1.0,
    max_diff_abs: Union[float, None] = None
  ) -> np.ndarray:
  """Interpolate outliers in an otherwise linear sequence.
  
  - For example: Measurement timestamps
  - I.e., goal is to make the sequence strictly increasing with approx. constant diff.
  
  Args:
    t: The sequence vals to fix. 1-dim.
    max_diff_rel: Maximum relative difference from regular linear increasing value [%]
      Used if `max_diff_abs` is None
    max_diff_abs: Maximum absolute difference from regular linear increasing value
      Used if not None 
  Returns:
    t: The interpolated sequence of strictly increasing vals. 1-dim.
  """
  assert isinstance(max_diff_rel, float)
  assert max_diff_abs is None or isinstance(max_diff_abs, float)
  from sklearn.linear_model import RANSACRegressor
  t = np.asarray(t)
  assert len(t.shape) == 1
  size = len(t)
  indices = np.arange(size)
  # Calculate max diff
  max_diff = max_diff_abs if max_diff_abs is not None else np.abs(np.median(np.diff(t)) * max_diff_rel)
  # Stage 1
  def interpolate_regression_outliers(t, max_diff):
    # Stage 1: Fit robust regression model
    reg = RANSACRegressor(random_state=0).fit(indices[:,np.newaxis], t)
    # Stage 1: Identify idxs for regression outliers
    t_preds = reg.predict(indices[:,np.newaxis])
    not_reg_outlier = np.abs(t_preds - t) < max_diff
    # Stage 1: Fix regression outliers
    f = interpolate.interp1d(indices[not_reg_outlier], t[not_reg_outlier],
      kind='linear', fill_value="extrapolate")
    return f(indices)
  t = interpolate_regression_outliers(t, max_diff=max_diff)
  # Stage 2
  def interpolate_non_strictly_increasing(t):
    not_si_outlier = np.concatenate([[True], t[1:] - t[:-1] > 0])
    f = interpolate.interp1d(indices[not_si_outlier], t[not_si_outlier],
      kind='linear', fill_value="extrapolate")
    return f(indices)
  while not np.logical_and.reduce(np.diff(t) > 0):
    t = interpolate_non_strictly_increasing(t)
  # Assert strictly increasing
  assert np.logical_and.reduce(np.diff(t) > 0)
  return t

def interpolate_data_outliers(
    x: np.ndarray,
    z_score: Union[int, float] = 3
  ) -> np.ndarray:
  """Recursively interpolate outliers in sensor data.

  - Example: ECG signal
  - Goal is to remove outliers in sensor data which may be caused by electrical interference etc.
  
  Args:
    vals: The signal data to interpolate. 1-dim.
    z_score: Significance score required for a data point to be interpolated
  Returns:
    vals: The interpolated signal data. 1-dim.
  """
  def interpolation_step(
      x: np.ndarray,
      z_score: Union[int, float]
    ) -> np.ndarray:
    x_z_score = stats.zscore(x)
    if np.isnan(x_z_score).all():
      return x
    not_outlier = np.abs(x_z_score) <= z_score
    indices = np.arange(len(x))
    interp = interpolate.interp1d(indices[not_outlier], x[not_outlier],
      kind='linear', fill_value='extrapolate')
    new_x = interp(indices)
    new_outlier = np.abs(stats.zscore(new_x)) > z_score
    if len(np.where(new_outlier)[0]) > 0:
      if new_outlier[0]:
        # Set first to mean if it is an outlier to avoid infinite recursion
        new_x[0] = np.mean(new_x[~new_outlier])
      if new_outlier[-1]:
        # Set last to mean if it is an outlier to avoid infinite recursion
        new_x[-1] = np.mean(new_x[~new_outlier])
      return interpolation_step(new_x, z_score)
    else:
      return new_x
  # Recursive interpolation of data outliers
  return interpolation_step(x=x, z_score=z_score)

def _component_periodicity(x: np.ndarray) -> list:
  """Compute the periodicity of the maximum frequency components

  Args:
    x: The signal data. Shape (n_sig, n_data)
  Returns:
    result: The periodicities. Shape (n_sig,)
  """
  x = np.asarray(x) # Make sure x is np array
  x = np.nan_to_num(x) # Replace NAs with 0
  assert x.ndim == 2, "x.ndim must equal 2"
  # Perform FFT
  w = fft.rfft(x, axis=1)
  # Determine maximum frequency component of each dim
  w_ = np.square(np.abs(w))
  w_ = div0(w_, np.sum(w_, axis=1)[:, np.newaxis], fill=0)
  idxs = np.argmax(w_, axis=1)
  # Compute periodicity for maximum frequency component
  return [w_[i,idx] for i, idx in enumerate(idxs)]

def select_most_periodic(x: np.ndarray) -> np.ndarray:
  """Select the most periodic signal

  Args:
    x: The signal data. Shape (n_sig, n_data)
  Returns:
    y: Signal with highest periodicity
  """
  x = np.asarray(x) # Make sure x is np array
  x = np.nan_to_num(x) # Replace NAs with 0
  assert x.ndim == 2, "x.ndim must equal 2"
  # Compute component periodicity
  p = _component_periodicity(x)
  idx = np.argmax(p)
  y = x[idx]
  assert x.shape[1] == y.shape[0]
  return y
