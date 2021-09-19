###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, September 2021              #
###############################################################################

import math
import numpy as np
from scipy import signal, interpolate
from scipy.sparse import spdiags
import scipy.ndimage.filters as ndif

from propy.stride_tricks import window_view, resolve_1d_window_view

def standardize(x, axis=-1):
  """Perform standardization
  Args:
    x: The input data
    axis: Axis over which to standardize
  Returns:
    x: The standardized data
  """
  x = np.asarray(x)
  x -= np.mean(x, axis=axis, keepdims=x.ndim>0)
  x /= np.std(x, axis=axis, keepdims=x.ndim>0)
  return x

def moving_average(x, size, axis=-1, pad_method='reflect'):
  """Perform moving average
  Args:
    x: The input data
    size: The size of the moving average window
    axis: Axis over which to calculate moving average
    pad_method: Method for padding ends to keep same dims
  Returns:
    x: The averaged data
  """
  x = np.array(x)
  if np.isnan(x).any():
    return x
  scaling_factor = 1000000000
  x_scaled = x * scaling_factor
  y_scaled = ndif.uniform_filter1d(
    x_scaled, size, mode=pad_method, origin=0, axis=axis)
  y = y_scaled / scaling_factor
  return y

def moving_average_size_for_response(sampling_freq, cutoff_freq):
  """Estimate the required moving average size to achieve a given response
  Args:
    sampling_freq: The sampling frequency [Hz]
    cutoff_freq: The desired cutoff frequency [Hz]
  Returns:
    size: The estimated moving average size
  """
  # Adapted from https://dsp.stackexchange.com/a/14648
  # cutoff freq in Hz
  F = cutoff_freq / sampling_freq
  size = int(math.sqrt(0.196202 + F * F) / F)
  #size = max(math.floor(size / 2.) * 2 + 1, 1)
  return max(size, 1)

def moving_std(x, size, overlap, fill_method='mean'):
  """Compute moving standard deviation
  Args:
    x: The data to be computed
    size: The size of the moving window
    overlap: The overlap of the moving windows
    fill_method: Method to fill the edges
  Returns:
    std: The moving standard deviations
  """
  x = np.array(x)
  x_view, _, pad_end = window_view(
    x=x,
    min_window_size=size,
    max_window_size=size,
    overlap=overlap)
  y_view = np.std(x_view, axis=-1)
  y = resolve_1d_window_view(y_view, size, overlap, pad_end, fill_method)
  return y

def detrend(z, Lambda, axis=-1):
  """Vectorized implementation of the detrending method by
    Tarvainen et al. (2002). Based on code listing in the Appendix.
  Args:
    z: The input signal
    Lambda: The lambda parameter
    axis: The axis along which should be detrended
  Returns:
    proc_z: The detrended signal
  """
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

def butter_bandpass(data, lowcut, highcut, fs, axis=-1, order=5):
  """Apply a butterworth bandpass filter.
  Args:
    data: The signal data
    lowcut: The lower cutoff frequency
    highcut: The higher cutoff frequency
    fs: The sampling frequency
    axis: The axis along which to apply the filter
    order: The order of the filter
  Returns:
    y: The filtered signal data
  """
  def butter_bandpass_filter(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a
  b, a = butter_bandpass_filter(lowcut, highcut, fs, order=order)
  y = signal.lfilter(b, a, data, axis=axis)
  return y

def estimate_freq(x, sampling_freq, axis=-1):
  """Use a fourier transform to determine maximum frequencies.
  Args:
    x: The signal data
    sampling_freq: The sampling frequency
    axis: The axis along which to estimate frequencies
  Returns:
    freq_hz: The maximum frequency [Hz]
  """
  # Compute the fourier transform
  x = np.asarray(x)
  # Change to 2-dim array if necessary
  if len(x.shape) == 1:
    x = np.expand_dims(x, axis=0)
  # Run the fourier transform
  w = np.fft.fft(x, axis=axis)
  freqs = np.fft.fftfreq(x.shape[axis])
  # Determine maximum frequency component
  idx = np.argmax(np.abs(w[:,1:w.shape[axis]//2]), axis=axis) + 1
  # Derive frequency in Hz
  freq_hz = abs(freqs[idx] * sampling_freq)
  # Squeeze if necessary
  freq_hz = np.squeeze(freq_hz)
  # Return
  return freq_hz

def estimate_freq_at_f_res(x, f_s, f_res, axis=-1):
  """Use a periodigram to estimate maximum frequencies at f_res.
  Args:
    x: The signal data
    f_s: The sampling frequency
    f_res: The desired frequency resolution
    axis: The axis along which to estimate frequencies
  Returns:
    freq_hz: The maximum frequency [Hz]
  """
  # Compute the fourier transform
  x = np.asarray(x)
  # Change to 2-dim array if necessary
  if len(x.shape) == 1:
    x = np.expand_dims(x, axis=0)
  # Determine the length of the fft
  n = f_s // f_res
  # Compute
  f, pxx = signal.periodogram(x, fs=f_s, nfft=n, detrend=False, axis=axis)
  #f, pxx = signal.welch(x, fs=f_s, window='boxcar', nperseg=n, noverlap=0,
  #  nfft=None, detrend='constant')
  # Return the maximum freq
  return f[np.argmax(pxx, axis=axis)]

def interpolate_cubic_spline(x, y, xs):
  """Interpolate data with a cubic spline.
  Args:
    x: The x values of the data we want to interpolate
    y: The y values of the data we want to interpolate
    xs: The x values at which we want to interpolate
  Returns:
    ys: The interpolated y values
  """
  x = np.nan_to_num(x) # Replace NAs with 0
  y = np.nan_to_num(y) # Replace NAs with 0
  cs = interpolate.CubicSpline(x, y, axis=1)
  return cs(xs)

def component_periodicity(x, axis=-1):
  """Compute the periodicity of the maximum frequency components
  Args:
    x: The signal data
    axis: The axis over which to compute perdiodicities
  Returns:
    result: The periodicities
  """
  axis = 1 if axis == -1 else axis
  x = np.asarray(x) # Make sure x is np array
  x = np.nan_to_num(x) # Replace NAs with 0
  assert x.ndim == 2, "x.ndim must equal 2"
  if axis == 0: x = np.transpose(x)
  # Perform FFT
  w = np.fft.fft(x, axis=1)
  # Determine maximum frequency component of each dim
  w_ = np.square(np.abs(w[:,0:w.shape[1]//2]))
  w_ = w_ / np.sum(w_, axis=1)[:, np.newaxis]
  idxs = np.argmax(w_, axis=1)
  # Compute periodicity for maximum frequency component
  return [w_[i,idx] for i, idx in enumerate(idxs)]

def select_most_periodic(x, axis=-1):
  """Select the most periodic signal
  Args:
    x: The 2-d signal data
    axis: The axis to reduce by selecting index with highest periodicity
  Returns:
    y: Signal with highest periodicity
  """
  axis = 1 if axis == -1 else axis
  x = np.asarray(x) # Make sure x is np array
  x = np.nan_to_num(x) # Replace NAs with 0
  assert x.ndim == 2, "x.ndim must equal 2"
  if axis == 0: x = np.transpose(x)
  # Compute component periodicity
  p = component_periodicity(x, 1)
  idx = np.argmax(p)
  y = x[idx]
  assert x.shape[1] == y.shape[0]
  return y

def windowed_standardize(x, window_size, windowed_mean=True, windowed_std=True):
  """Perform dynamic standardization based on windowed mean and std
  Args:
    x: The input data
    window_size: The size of the moving window
    windowed_mean: Boolean indicating whether mean should be windowed
    windowed_std: Boolean indicating whether std should be windowed
  Returns:
    y: The standardized data
  """
  x = np.asarray(x)
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
