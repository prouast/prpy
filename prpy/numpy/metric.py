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

from __future__ import annotations

import numpy as np
from scipy import signal
from typing import Callable, Union

MAX_SHIFT_FIX = 30 # samples

def _to_np(x: Union[np.ndarray, float, int, object]) -> np.ndarray:
  if isinstance(x, np.ndarray):
    return x
  numpy_meth = getattr(x, "numpy", None)
  if callable(numpy_meth):
    try:
      return numpy_meth()
    except Exception:
      pass
  return np.asarray(x)

def _ensure_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
  if x.ndim == 1:
    return x[None, :], True
  return x, False

def _masked_sum(x: np.ndarray, mask: np.ndarray, axis: int, *, keepdims=False):
  return np.where(mask, x, 0.0).sum(axis=axis, keepdims=keepdims)

def _safe_mean(x: np.ndarray, mask: np.ndarray, axis: int):
  n_valid = mask.sum(axis=axis, keepdims=True)
  n_safe = np.where(n_valid == 0, 1, n_valid)
  return _masked_sum(x, mask, axis, keepdims=True) / n_safe

def _mag2db(mag: Union[np.ndarray, np.float64]) -> np.ndarray:
  """
  Magnitude to decibels element-wise.

  Args:
    mag: Magnitude. Arbitrary shape.
  Returns:
    out: Decibels. Same shape as input.
  """
  return 20. * np.log10(mag)

def _metric_reduce(res: np.ndarray, squeezed: bool):
  return res.squeeze(0) if squeezed else res

def mae(
    y_true: Union[np.ndarray, object],
    y_pred: Union[np.ndarray, object],
    *,
    ignore_nan: bool = False,
    **kw
  ) -> np.ndarray:
  """
  Mean absolute error

  Args:
    y_true: True values. Shape 1D (N,) or 2D (B, N)
    y_pred: Predicted values. Same shape as `y_true`.
    ignore_nan: If true, metric is computed on non-NaN elements only.
  Returns:
    mae: The mean absolute error. Shape () or (B,)
  """
  y_true = _to_np(y_true)
  y_pred = _to_np(y_pred)
  y_true, squeezed = _ensure_2d(y_true)
  y_pred, _ = _ensure_2d(y_pred)
  assert y_true.shape == y_pred.shape
  err = np.abs(y_true - y_pred)
  if not ignore_nan:
    res = err.mean(axis=1)
  else:
    mask = np.isfinite(err)
    n_valid = mask.sum(axis=1)
    err_sum = _masked_sum(err, mask, axis=1)
    res = np.where(n_valid > 0, err_sum / n_valid, np.nan)
  return _metric_reduce(res, squeezed)

def mse(
    y_true: Union[np.ndarray, object],
    y_pred: Union[np.ndarray, object],
    *,
    ignore_nan: bool = False,
    **kw
  ) -> np.ndarray:
  """
  Mean squared error

  Args:
    y_true: True values. Shape 1D (N,) or 2D (B, N)
    y_pred: Predicted values. Same shape as `y_true`.
    ignore_nan: If true, metric is computed on non-NaN elements only.
  Returns:
    mse: The mean squared error. Shape () or (B,)
  """
  y_true = _to_np(y_true)
  y_pred = _to_np(y_pred)
  y_true, squeezed = _ensure_2d(y_true)
  y_pred, _ = _ensure_2d(y_pred)
  sq = (y_true - y_pred) ** 2
  if not ignore_nan:
    res = sq.mean(axis=1)
  else:
    mask = np.isfinite(sq)
    n_valid = mask.sum(axis=1)
    sq_sum = _masked_sum(sq, mask, axis=1)
    res = np.where(n_valid > 0, sq_sum / n_valid, np.nan)
  return _metric_reduce(res, squeezed)

def rmse(
    y_true: Union[np.ndarray, object],
    y_pred: Union[np.ndarray, object],
    *,
    ignore_nan: bool = False,
    **kw
  ) -> np.ndarray:
  """
  Root mean squared error

  Args:
    y_true: True values. Shape 1D (N,) or 2D (B, N)
    y_pred: Predicted values. Same shape as `y_true`.
    ignore_nan: If true, metric is computed on non-NaN elements only.
  Returns:
    rmse: The root mean squared error. Shape () or (B,)
  """
  return np.sqrt(mse(y_true, y_pred, ignore_nan=ignore_nan))

def cor(
    y_true: Union[np.ndarray, object],
    y_pred: Union[np.ndarray, object],
    *,
    ignore_nan: bool = False,
    **kw
  ) -> np.ndarray:
  """
  Pearson's correlation coefficient

  Args:
    y_true: True values. Shape 1D (N,) or 2D (B, N)
    y_pred: Predicted values. Same shape as `y_true`.
    ignore_nan: If true, metric is computed on non-NaN elements only.
  Returns:
    cor: The correlation coefficients. Shape TODO
  """
  y_true = _to_np(y_true)
  y_pred = _to_np(y_pred)
  y_true, squeezed = _ensure_2d(y_true)
  y_pred, _ = _ensure_2d(y_pred)
  assert y_true.shape == y_pred.shape
  mask = np.isfinite(y_true) & np.isfinite(y_pred) if ignore_nan else np.ones_like(y_true, bool)
  mean_t = _safe_mean(y_true, mask, axis=1)
  mean_p = _safe_mean(y_pred, mask, axis=1)
  r_true = np.where(mask, y_true - mean_t, 0.0)
  r_pred = np.where(mask, y_pred - mean_p, 0.0)
  n_valid = mask.sum(axis=1)
  cov = (r_true * r_pred).sum(axis=1) / np.where(n_valid == 0, 1, n_valid)
  var_t = (r_true ** 2).sum(axis=1) / np.where(n_valid == 0, 1, n_valid)
  var_p = (r_pred ** 2).sum(axis=1) / np.where(n_valid == 0, 1, n_valid)
  res = cov / (np.sqrt(var_t) * np.sqrt(var_p))
  res = np.where(n_valid > 0, res, np.nan)
  return _metric_reduce(res, squeezed)

def snr(
    f_true: Union[np.ndarray, float, int, object],
    y_pred: Union[np.ndarray, object],
    f_s: Union[float, int],
    f_res: float,
    *,
    tol: Union[float, int] = .1,
    f_min: Union[float, int] = .5,
    f_max: Union[float, int] = 4.
  ) -> np.ndarray:
  """
  Signal-to-noise ratio
  
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
  assert isinstance(f_s, (float, int, np.float32, np.float64, np.int32, np.int64)), f"f_s should be float or int but was {type(f_s)}"
  assert isinstance(f_res, (float, int, np.float32, np.float64, np.int32, np.int64)), f"f_res should be float or int but was {type(f_res)}"
  assert isinstance(tol, (float, int, np.float32, np.float64, np.int32, np.int64)), f"tol should be float or int but was {type(tol)}"
  assert isinstance(f_min, (float, int, np.float32, np.float64, np.int32, np.int64)), f"f_min should be float or int but was {type(f_min)}"
  assert isinstance(f_max, (float, int, np.float32, np.float64, np.int32, np.int64)), f"f_max should be float or int but was {type(f_max)}"
  f_s = float(f_s)
  f_res = float(f_res)
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
    snr = _mag2db(s_power / (all_power - s_power))
    return np.squeeze(snr)
  else:
    return np.full(y_pred.shape[:-1], fill_value=np.nan, dtype=y_pred.dtype)

def make_shift_pooled_metric(
    base_metric: Callable[[np.ndarray, np.ndarray, np.ndarray, float, float], np.ndarray],
    *,
    maximise: bool = False,
    max_shift_t: float = 0.3,
    max_shift_fix: int = MAX_SHIFT_FIX,
  ) -> Callable[[np.ndarray, np.ndarray, np.ndarray, float, float], np.ndarray]:
  """Create a shift-pooled variant of any batch metric.
  
  Args:
    base_metric: Callable metric.
    maximise: If True, choose the maximum value across shifts; otherwise choose the minimum.
    max_shift_t: Maximum absolute shift considered (seconds).
    max_shift_fix: Hard cap for the shift in samples regardless of the sampling frequency.
  Returns:
   metric_sp: Callable with the same signature as base_metric that performs the
     shift-pooled evaluation.
  """
  shifts_all = np.arange(-max_shift_fix, max_shift_fix + 1, dtype=int)
  S = shifts_all.size
  sign = -1.0 if maximise else +1.0
  def metric_valid(y_true, y_pred, f_s, f_min, f_max):
    y_true = _to_np(y_true)
    y_pred = _to_np(y_pred)
    f_s = _to_np(f_s)
    y_true, squeezed = _ensure_2d(y_true)
    y_pred, _ = _ensure_2d(y_pred)
    if f_s.ndim == 0:
      f_s = np.full((y_true.shape[0],), float(f_s), dtype=float)
    B, T = y_true.shape
    rolled = np.stack([np.roll(y_pred, s, axis=1) for s in shifts_all], axis=0)
    idx = np.arange(T)
    mask = np.empty((S, T), bool)
    for i, s in enumerate(shifts_all):
      mask[i] = idx >= s if s >= 0 else idx < (T + s)
    vals = np.empty((S, B), dtype=float)
    for i, s in enumerate(shifts_all):
      m = mask[i]
      if not m.any():
        vals[i] = np.inf
        continue
      y_t_slice = y_true[:, m]
      y_p_slice = rolled[i, :, m].T
      vals[i] = base_metric(y_t_slice, y_p_slice, f_s=f_s, f_min=f_min, f_max=f_max)
    k_per_item = np.minimum(np.floor(max_shift_t * f_s).astype(int), max_shift_fix)
    admissible = np.abs(shifts_all)[:, None] <= k_per_item[None, :]
    penalised = np.where(admissible, sign*vals, np.inf)
    best = penalised.min(axis=0)
    res = (sign * best).astype(np.float32)
    return res[0] if squeezed else res
  return metric_valid

cor_shift = make_shift_pooled_metric(lambda t, p, **_: cor(t, p), maximise=True)
mae_shift = make_shift_pooled_metric(lambda t, p, **_: mae(t, p))
mse_shift = make_shift_pooled_metric(lambda t, p, **_: mse(t, p))
rmse_shift = make_shift_pooled_metric(lambda t, p, **_: rmse(t, p))
