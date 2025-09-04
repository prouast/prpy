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
from typing import Union, Any

from prpy.numpy.core import div0
from prpy.numpy.interp import interpolate_skipped

def estimate_freq(
    x: np.ndarray,
    f_s: Union[float, int],
    f_range: tuple = None,
    f_res: float = None,
    method: str = 'fft',
    interp_skipped: bool = False,
    axis: int = -1,
    keepdims: bool = False,
    **kw: Any
  ) -> np.ndarray:
  """
  Determine maximum frequencies in x.

  Args:
    x: The signal data. Shape: (n_data,) or (n_sig, n_data)
    f_s: The sampling frequency [Hz]
    f_range: Optional expected range of freqs [Hz] - (min, max)
    f_res: Optional frequency resolution for analysis [Hz]
      (useful if signal small; applies only to periodogram)
    method: The method to be used [fft, peak, or periodogram]
    interp_skipped: Insert interpolated detection for presumably skipped events
    axis: The axis along which to estimate frequencies
    keepdims: Keep squeezable first dim in x
    **kw: Extra args forwarded to the underlying frequency estimator.
  Returns:
    f_out: The maximum frequencies [Hz]. Shape: (n_sig,)
  """
  assert isinstance(method, str)
  if method == 'fft':
    return estimate_freq_fft(x, f_s=f_s, f_range=f_range, axis=axis, keepdims=keepdims)
  elif method == 'peak':
    kw.update(f_res=f_res)
    return estimate_freq_peak(x, f_s=f_s, f_range=f_range, interp_skipped=interp_skipped, axis=axis, keepdims=keepdims, **kw)
  elif method == 'periodogram':
    return estimate_freq_periodogram(x, f_s=f_s, f_range=f_range, f_res=f_res, axis=axis, keepdims=keepdims)
  else:
    return ValueError(f"method should be 'peak', 'fft', or 'periodogram' but was {method}")

def estimate_freq_fft(
    x: np.ndarray,
    f_s: Union[float, int],
    f_range: Union[tuple, None] = None,
    axis: int = -1,
    keepdims: bool = False,
  ) -> np.ndarray:
  """
  Use a fourier transform to determine maximum frequencies.
  
  Args:
    x: The signal data. Shape: (n_data,) or (n_sig, n_data)
    f_s: The sampling frequency [Hz]
    f_range: Optional expected range of freqs [Hz] - (min, max)
    axis: The axis along which to estimate frequencies
    keepdims: Keep squeezable first dim in x
  Returns:
    f_out: The maximum frequencies [Hz]. Shape: (n_sig,)
  """
  assert isinstance(f_s, (float, int)) and f_s > 0
  assert f_range is None or (isinstance(f_range, tuple) and len(f_range) == 2 and all(isinstance(i, (int, float)) for i in f_range))
  assert isinstance(axis, int) and (axis == 0 or axis == 1 or axis == -1)
  x = np.asarray(x)
  onedim_input = len(x.shape) == 1
  # Change to 2-dim array if necessary
  if onedim_input:
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
  if onedim_input or not keepdims:
    f_out = np.squeeze(f_out)
  # Return
  return f_out

def estimate_freq_peak(
    x: np.ndarray,
    f_s: Union[float, int],
    f_range: Union[tuple, None] = None,
    interp_skipped: bool = False,
    axis: int = -1,
    keepdims: bool = False,
    **kw: Any
  ) -> np.ndarray:
  """
  Use peak detection to determine maximum frequencies in x.

  Args:
    x: The signal data. Shape: (n_data,) or (n_sig, n_data)
    f_s: The sampling frequency [Hz]
    f_range: Optional expected range of freqs [Hz] - (min, max)
    interp_skipped: Insert interpolated detection for presumably skipped events
    axis: The axis along which to estimate frequencies,
    keepdims: Keep squeezable first dim in x
    **detect_kwargs: Extra args forwarded to `detect_valid_peaks`.
  Returns:
    f_out: The maximum frequencies [Hz]. Shape: (n_sig,)
  """
  from prpy.numpy.detect import detect_valid_peaks
  assert isinstance(f_s, (float, int)) and f_s > 0
  assert f_range is None or (isinstance(f_range, tuple) and len(f_range) == 2 and all(isinstance(i, (int, float)) for i in f_range))
  assert isinstance(axis, int) and (axis == 0 or axis == 1 or axis == -1)
  x = np.asarray(x)
  onedim_input = len(x.shape) == 1
  # Change to 2-dim array if necessary
  if onedim_input:
    x = np.expand_dims(x, axis=0)
  # Helper for a single 1-D trace
  def _freq_1d(trace: np.ndarray):
    # Find peaks in the signal
    seqs, _ = detect_valid_peaks(
      vals=trace,
      f_s=f_s,
      f_range=f_range,
      **kw
    )
    if not seqs: return np.nan  
    diffs_list = [np.diff(seq) for seq in seqs if len(seq) > 1]
    if not diffs_list: return np.nan
    diffs = np.concatenate(diffs_list)
    if interp_skipped and diffs.size:
      diffs = interpolate_skipped(diffs, threshold=0.3)
    if diffs.size == 0:
      # Not enough peaks
      return np.nan
    # Derive the frequency
    return f_s/np.median(diffs)
  # Apply function
  f_out = np.apply_along_axis(_freq_1d, axis=axis, arr=x)
  # Squeeze if necessary
  if onedim_input or not keepdims:
    f_out = np.squeeze(f_out)
  # Return
  return f_out

def estimate_freq_periodogram(
    x: np.ndarray,
    f_s: Union[float, int],
    f_range: Union[tuple, None] = None,
    f_res: Union[float, None] = None,
    axis: int = -1,
    keepdims: bool = False
  ) -> np.ndarray:
  """
  Use a periodigram to estimate maximum frequencies at f_res.
  
  - When signal is sampled at a lower frequency than f_res, this is essentially done
    by interpolating in the frequency domain.
  
  Args:
    x: The signal data. Shape: (n_data,) or (n_sig, n_data)
    f_s: The sampling frequency [Hz]
    f_range: Optional expected range of freqs [Hz] - (min, max)
    f_res: Optional frequency resolution for analysis [Hz]
    axis: The axis along which to estimate frequencies
    keepdims: Keep squeezable first dim in x
  Returns:
    f_out: The maximum frequencies [Hz]. Shape: (n_sig,)
  """
  assert isinstance(f_s, (float, int, np.float32, np.float64, np.int32, np.int64)) and f_s > 0
  assert f_range is None or (isinstance(f_range, tuple) and len(f_range) == 2 and all(isinstance(i, (int, float)) for i in f_range))
  assert f_res is None or (isinstance(f_res, (float, int)) and f_res > 0)
  assert isinstance(axis, int) and (axis == 0 or axis == 1 or axis == -1)
  x = np.asarray(x)
  onedim_input = len(x.shape) == 1
  # Change to 2-dim array if necessary
  if onedim_input:
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
  if onedim_input or not keepdims:
    f_out = np.squeeze(f_out)
  # Return
  return f_out

def _component_periodicity(x: np.ndarray) -> list:
  """
  Compute the periodicity of the maximum frequency components

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
  """
  Select the most periodic signal

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
