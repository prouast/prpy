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
from scipy import signal, fft
from typing import Union

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
    return ValueError(f"method should be 'peak', 'fft', or 'periodogram' but was {method}")

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
  # TODO: Use `detect_valid_peaks`?
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
  assert isinstance(f_s, (float, int, np.float32, np.float64, np.int32, np.int64)) and f_s > 0
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
