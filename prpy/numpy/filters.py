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

import math
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.sparse import spdiags
from typing import Union, Tuple

from prpy.numpy.stride_tricks import window_view, resolve_1d_window_view

def moving_average(
    x: np.ndarray,
    size: int,
    axis: int = -1,
    mode: str = 'reflect',
    center: bool = True,
    precision = np.float64,
  ) -> np.ndarray:
  """NaN-aware moving average.

  Args:
    x: The input data
    size: The size of the moving average window
    axis: Axis over which to calculate moving average
    mode: Padding mode for the edges (reflect, nearest, wrap, ...)
    center: True -> Window is centered; False -> Left-aligned
    precision: working dtype (default float64; use np.float128 for extreme cases)
  Returns:
    y: The averaged data
  """
  if not isinstance(size, int) or size < 1: raise ValueError('`size` must be a positive integer')
  if not isinstance(axis, int): raise TypeError('`axis` must be an int')
  
  # Promote to the chosen working precision
  work = np.asarray(x, dtype=precision)
  out_dtype = x.dtype if isinstance(x, np.ndarray) else precision
  origin = 0 if center else size // 2
  
  filled = np.nan_to_num(work, nan=0.0)
  valid = np.isfinite(work).astype(precision)

  win_sum = uniform_filter1d(filled, size, axis=axis, mode=mode, origin=origin)
  win_count = uniform_filter1d(valid, size, axis=axis, mode=mode, origin=origin)

  with np.errstate(divide='ignore', invalid='ignore'):
    out = win_sum / win_count
  out[win_count == 0] = np.nan
  
  return out.astype(out_dtype, copy=False)

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
  x_view, pad_start, pad_end = window_view(
    x=x,
    min_window_size=size,
    max_window_size=size,
    overlap=overlap)
  y_view = np.std(x_view, axis=-1)
  y = resolve_1d_window_view(
    x=y_view,
    window_size=size,
    overlap=overlap,
    pad_start=pad_start,
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
  A = I + (Lambda**2) * np.dot(D2.T, D2)
  # Compute the detrending operation (vectorized)
  if axis == 0:
    z = np.transpose(z)
  z_trend = np.linalg.solve(A, z.T)
  proc_z = (z.T - z_trend)
  if axis == 1:
    proc_z = np.transpose(proc_z)
  # Squeeze if necessary
  proc_z = np.squeeze(proc_z)
  # Return
  return proc_z

def detrend_frequency_response(
    size: int,
    Lambda: int,
    f_s: float
  ) -> float:
  """
  Return the estimated frequency response for Tarvainen et al. (2002) in Hz
  
  Args:
    size: The size of the signal
    Lambda: The Lambda parameter used
    f_s: The sampling frequency of the signal
  Returns:
    The estimated frequency response in Hz
  """
  T = size
  # Identity matrix
  I = np.identity(T)
  # Regularization matrix
  D2 = spdiags(
    [np.ones(T), -2*np.ones(T), np.ones(T)],
    [0, 1, 2], (T-2), T).toarray()
  # Inverse of I+lambda^2*D2’*D2
  inv = np.linalg.inv(I + (Lambda**2) * np.dot(D2.T, D2))
  # The time-varying FIR high pass filter
  L = (I - inv)
  # Compute the fourier transform of the middle row of L
  # Take the magnitude of the frequency response
  L_freq = np.abs(np.fft.fft(L[T//2]))[:T//2]
  # FFT frequencies in Hz
  freqs = np.fft.fftfreq(T, d=1/f_s)[0:T//2]
  # The frequency in Hz
  freq = freqs[np.argmax(L_freq > 0.64)]
  # Return
  return float(freq)

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
  y = signal.filtfilt(b=b, a=a, x=x, axis=axis)
  return y

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
    mean = moving_average(x, size=window_size)
  else:
    mean = np.mean(x)
  if windowed_std:
    std = moving_std(x, size=window_size, overlap=window_size-1)
  else:
    std = np.std(x)
  x -= mean
  x /= std
  return x
