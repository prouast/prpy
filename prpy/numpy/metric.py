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
from scipy import signal
from typing import Union

def mag2db(mag: Union[np.ndarray, np.float64]) -> np.ndarray:
  """Magnitude to decibels element-wise.

  Args:
    mag: Magnitude. Arbitrary shape.
  Returns:
    out: Decibels. Same shape as input.
  """
  assert isinstance(mag, (np.ndarray, np.float64))
  return 20. * np.log10(mag)

def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis: Union[int, None] = -1
  ) -> np.ndarray:
  """Mean absolute error

  Args:
    y_true: True values. Shape (..., dim_n, dim_axis)
    y_pred: Predicted values. Shape (..., dim_n, dim_axis)
    axis: Axis along which the means are computed
  Returns:
    mae: The mean absolute error. Shape (..., dim_n)
  """
  assert axis is None or isinstance(axis, int)
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  assert y_true.shape == y_pred.shape
  if np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
    return np.mean(np.abs(y_true - y_pred), axis=axis)
  else:
    return np.full(y_true.shape[:axis], fill_value=np.nan, dtype=y_true.dtype)

def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis: Union[int, None] = -1
  ) -> np.ndarray:
  """Mean squared error

  Args:
    y_true: True values. Shape (..., dim_n, dim_axis)
    y_pred: Predicted values. Shape (..., dim_n, dim_axis)
    axis: Axis along which the means are computed
  Returns:
    mse: The mean squared error. Shape (..., dim_n)
  """
  assert axis is None or isinstance(axis, int)
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  assert y_true.shape == y_pred.shape
  if np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
    return np.mean(np.square(y_true - y_pred), axis=axis)
  else:
    return np.full(y_true.shape[:axis], fill_value=np.nan, dtype=y_true.dtype)

def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis: Union[int, None] = -1
  ) -> np.ndarray:
  """Root mean squared error

  Args:
    y_true: True values. Shape (..., dim_n, dim_axis)
    y_pred: Predicted values. Shape (..., dim_n, dim_axis)
    axis: Axis along which the means are computed
  Returns:
    rmse: The root mean squared error. Shape (..., dim_n)
  """
  assert axis is None or isinstance(axis, int)
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  assert y_true.shape == y_pred.shape
  if np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
    return np.sqrt(np.mean(np.square(y_true - y_pred), axis=axis))
  else:
    return np.full(y_true.shape[:axis], fill_value=np.nan, dtype=y_true.dtype)

def cor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis: Union[int, None] = -1
  ) -> np.ndarray:
  """Pearson's correlation coefficient

  Args:
    y_true: True values. Shape (..., dim_n, dim_axis)
    y_pred: Predicted values. Shape (..., dim_n, dim_axis)
    axis: Axis along which correlations are computed
  Returns:
    cor: The correlation coefficients. Shape (..., dim_n)
  """
  assert axis is None or isinstance(axis, int)
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  assert y_true.shape == y_pred.shape
  if np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
    res_true = y_true - np.mean(y_true, axis=axis, keepdims=True)
    res_pred = y_pred - np.mean(y_pred, axis=axis, keepdims=True)
    cov = np.mean(res_true * res_pred, axis=axis)
    var_true = np.mean(res_true**2, axis=axis)
    var_pred = np.mean(res_pred**2, axis=axis)
    sigma_true = np.sqrt(var_true)
    sigma_pred = np.sqrt(var_pred)
    return cov / (sigma_true * sigma_pred)
  else:
    return np.full(y_true.shape[:axis], fill_value=np.nan, dtype=y_true.dtype)

def snr(
    f_true: np.ndarray,
    y_pred: np.ndarray,
    f_s: Union[float, int],
    f_res: float,
    tol: Union[float, int] = .1,
    f_min: Union[float, int] = .5,
    f_max: Union[float, int] = 4.
  ) -> np.ndarray:
  """Signal-to-noise ratio
  Args:
    f_true: The true frequencies. Shape (b,) or ()
    y_pred: Predicted vals. Shape (b, t) or (t,)
    f_s: Sampling frequency
    f_res: Frequency resolution
    tol: Frequency domain tolerance
    f_min: Minimum frequency included in metric calculation
    f_max: Maximum frequency included in metric calculation
  Returns:
    snr: The signal to noise ratio. Shape (b,) or ()
  """
  assert isinstance(f_s, (float, int))
  assert isinstance(f_res, float)
  assert isinstance(tol, (float, int))
  assert isinstance(f_min, (float, int))
  assert isinstance(f_max, (float, int))
  f_s = float(f_s)
  tol = float(tol)
  f_min = float(f_min)
  f_max = float(f_max)
  f_true = np.asarray(f_true)
  y_pred = np.asarray(y_pred)
  assert len(y_pred.shape) == 1 or len(y_pred.shape) == 2
  assert f_true.shape == y_pred.shape[:-1]
  if np.all(np.isfinite(y_pred)) and np.all(np.isfinite(f_true)):
    n = f_s // f_res
    f, pxx = signal.periodogram(y_pred, fs=f_s, nfft=n, detrend=False, axis=-1)
    if len(y_pred.shape) == 2:
      f = np.broadcast_to(f[np.newaxis], pxx.shape)
      f_true = f_true[...,np.newaxis]
    gt_mask_1 = (f >= f_true - tol) & (f <= f_true + tol)
    gt_mask_2 = (f >= f_true * 2 - tol) & (f <= f_true * 2 + tol)
    gt_mask = gt_mask_1 | gt_mask_2
    s_power = np.sum(pxx * gt_mask, axis=-1)
    f_mask = (f >= f_min) & (f <= f_max)
    all_power = np.sum(pxx * f_mask, axis=-1)
    snr = mag2db(s_power / (all_power - s_power))
    return np.squeeze(snr)
  else:
    return np.full(y_pred.shape[:-1], fill_value=np.nan, dtype=y_pred.dtype)
