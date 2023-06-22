###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, September 2021              #
###############################################################################

import numpy as np
from scipy import signal

def mag2db(mag):
  """Magnitude to decibels element-wise.
  Args:
    mag: Magnitude. np.ndarray with arbitrary shape.
  Returns:
    out: Decibels. Same shape as input.
  """
  return 20. * np.log10(mag)

def mae(y_true, y_pred, axis=-1):
  """Mean absolute error
  Args:
    y_true: True values. Shape (..., dim_n, dim_axis)
    y_pred: Predicted values. Shape (..., dim_n, dim_axis)
    axis: Axis along which the means are computed
  Returns:
    mae: The mean absolute error. Shape (..., dim_n)
  """
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  assert y_true.shape == y_pred.shape
  if np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
    return np.mean(np.abs(y_true - y_pred), axis=axis)
  else:
    return np.full(y_true.shape[:axis], fill_value=np.nan, dtype=y_true.dtype)

def mse(y_true, y_pred, axis=-1):
  """Mean squared error
  Args:
    y_true: True values. Shape (..., dim_n, dim_axis)
    y_pred: Predicted values. Shape (..., dim_n, dim_axis)
    axis: Axis along which the means are computed
  Returns:
    mse: The mean squared error. Shape (..., dim_n)
  """
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  assert y_true.shape == y_pred.shape
  if np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
    return np.mean(np.square(y_true - y_pred), axis=axis)
  else:
    return np.full(y_true.shape[:axis], fill_value=np.nan, dtype=y_true.dtype)

def rmse(y_true, y_pred, axis=-1):
  """Root mean squared error
  Args:
    y_true: True values. Shape (..., dim_n, dim_axis)
    y_pred: Predicted values. Shape (..., dim_n, dim_axis)
    axis: Axis along which the means are computed
  Returns:
    rmse: The root mean squared error. Shape (..., dim_n)
  """
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  assert y_true.shape == y_pred.shape
  if np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred)):
    return np.sqrt(np.mean(np.square(y_true - y_pred), axis=axis))
  else:
    return np.full(y_true.shape[:axis], fill_value=np.nan, dtype=y_true.dtype)

def cor(y_true, y_pred, axis=-1):
  """Pearson's correlation coefficient
  Args:
    y_true: True values. Shape (..., dim_n, dim_axis)
    y_pred: Predicted values. Shape (..., dim_n, dim_axis)
    axis: Axis along which the means are computed
  Returns:
    rmse: The correlation coefficients. Shape (..., dim_n)
  """
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

def snr(f_true, y_pred, f_s, f_res, tol=0.1, f_min=0.5, f_max=4):
  """Signal-to-noise ratio
  Args:
    f_true: The true frequencies. Shape (b,) or ()
    y_pred: Predicted vals. Shape (b, t) or (t,)
    f_s: Sampling frequency
    f_res: Frequency resolution
  Returns:
    snr: The signal to noise ratio. Shape (b,) or ()
  """
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
    s_power = np.sum(np.take(pxx, np.where(gt_mask_1 | gt_mask_2)), axis=-1)
    f_mask = (f >= f_min) & (f <= f_max)
    all_power = np.sum(np.take(pxx, np.where(f_mask)), axis=-1)
    snr = mag2db(s_power / (all_power - s_power))
    return np.squeeze(snr)
  else:
    return np.full(y_pred.shape[:-1], fill_value=np.nan, dtype=y_pred.dtype)
