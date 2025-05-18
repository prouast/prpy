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

import logging
import math
import numpy as np
from scipy import signal, interpolate, stats
from typing import Union, Callable

def interpolate_vals(
    x: np.ndarray,
    val_fn: Callable[[np.ndarray], np.ndarray] = lambda x: np.isnan(x)
  ) -> np.ndarray:
  """
  Linearly interpolate vals matching val_fn

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

def interpolate_filtered(
    t_in: np.ndarray,
    s_in: np.ndarray,
    t_out: np.ndarray,
    band: tuple = None,
    order: int = 4,
    extrapolate: bool = False,
    fill_nan: bool = True,
    axis: int = 0
  ) -> np.ndarray:
  """
  Interpolate data with bandpass filter and zero-phase polyphase resample (with PCHIP fallback) from `t_in` to `t_out`
  
  Args:
    t_in: The timestamp values we want to interpolate [seconds]. ndarray, shape (T,)
    s_in: The signal values we want to interpolate. ndarray, shape (..., T, ...)
    t_out: The new timestamp values at which we want to interpolate [seconds]. ndarray, shape (T2,)
    band: Optional (low, high) band tuple in Hz.
    order: Butterworth filter order, with higher values producing a steeper roll-off in the pass-band
    extrapolate: Whether to extrapolate to out-of-bounds points (only applies to PCHIP fallback)
    fill_nan: If True, any NaNs in `s_in` get linearly interpolated first (default=True)
    axis: Time axis of s_in.
  Returns:
    s_out: The interpolated signal values, shape (..., T2, ...)
  """
  t_in, s_in, t_out = map(np.asarray, (t_in, s_in, t_out))
  axis = axis % s_in.ndim
  def _nan_template(dt_len):
    shp = list(s_in.shape)
    shp[axis] = dt_len
    return np.full(shp, np.nan, dtype=s_in.dtype)
  def is_uniform(ts, tol=1e-6):
    dt = np.diff(ts)
    return len(dt) > 0 and np.allclose(dt, dt[0], atol=tol)
  uniform = is_uniform(t_in) and is_uniform(t_out)
  can_filter = isinstance(band, tuple) and uniform
  has_nan = np.isnan(s_in).any()
  # Optionally fill NaNs per time-series (handles 1-D or N-D with axis)
  if fill_nan:
    def _fill_1d(y):
      m = np.isnan(y)
      if m.any():
        y = y.copy()
        y[m] = np.interp(t_in[m], t_in[~m], y[~m])
      return y
    s_in = np.apply_along_axis(_fill_1d, axis, s_in)
  # Nothing to do if grids are identical
  if np.array_equal(t_in, t_out):
    return s_in
  # Mask in-range vs. out-of-range
  inside = (t_out >= t_in[0]) & (t_out <= t_in[-1])
  if not inside.any() and not (extrapolate and not can_filter):
    # no overlap at all - return all-nan
    return _nan_template(t_out.size)
  if can_filter and (fill_nan or not has_nan):
    # zero-phase Butterworth
    fs_in = 1.0 / np.median(np.diff(t_in))
    low, high = band
    nyq = 0.5 * fs_in
    b, a = signal.butter(order, [low/nyq, high/nyq], btype="band")
    s_filt = signal.filtfilt(b, a, s_in, axis=axis)
    # Polyphase resample - only resample the region we need
    t_mid = t_out[inside]
    fs_out = 1.0 / np.median(np.diff(t_mid))
    g = math.gcd(int(round(fs_in)), int(round(fs_out)))
    up, down = int(round(fs_out))//g, int(round(fs_in))//g
    s_rs = signal.resample_poly(s_filt, up, down, axis=axis)
    # Align length exactly to len(ts)
    if s_rs.shape[axis] != t_mid.size:
      slicer = [slice(None)] * s_rs.ndim
      slicer[axis] = slice(0, t_mid.size)
      s_rs = s_rs[tuple(slicer)]
    # inject into full-length nan canvas
    s_out = _nan_template(t_out.size)
    sl = [slice(None)] * s_out.ndim
    sl[axis] = inside
    s_out[tuple(sl)] = s_rs
    return s_out
  # Fallback: shape‑preserving spline (PCHIP)
  nan_mask = None
  if not fill_nan and has_nan:
    # Create 1D nan taking into account all extra dims in s
    other_axes = tuple(i for i in range(s_in.ndim) if i != axis)
    nan_mask = np.any(np.isnan(s_in), axis=other_axes)
  # Build PCHIP on the finite points
  t_good = t_in[~nan_mask] if nan_mask is not None else t_in
  y_good = np.take(s_in, np.nonzero(~nan_mask)[0], axis=axis) if nan_mask is not None else s_in
  # Bail out safely if we have < 2 good samples
  if t_good.size < 2:
    return _nan_template(t_out.size)
  pchip = interpolate.PchipInterpolator(t_good, y_good, axis=axis, extrapolate=extrapolate)
  if extrapolate:
    # full call, no NaN padding needed
    s_out = pchip(t_out)
  else:
    # only compute inside region, leave outside as NaN
    t_mid = t_out[inside]
    s_mid = pchip(t_mid)
    s_out = _nan_template(t_out.size)
    sl = [slice(None)] * s_out.ndim
    sl[axis] = inside
    s_out[tuple(sl)] = s_mid
  s_out = pchip(t_out)
  # If propagating nans, carve them back
  if nan_mask is not None:
    prev = np.concatenate(([False], nan_mask[:-1]))
    nxt = np.concatenate((nan_mask[1:], [False]))
    starts = np.nonzero(nan_mask & ~prev)[0]
    ends = np.nonzero(nan_mask & ~nxt)[0]
    if starts.size:
      t0 = t_in[starts][:, None]
      t1 = t_in[ends][:, None]
      inside = np.any((t_out >= t0) & (t_out <= t1), axis=0)
      sl = [slice(None)] * s_out.ndim
      sl[axis] = inside
      s_out[tuple(sl)] = np.nan
  return s_out

def interpolate_linear_sequence_outliers(
    t: np.ndarray,
    max_diff_rel: float = 1.0,
    max_diff_abs: Union[float, None] = None
  ) -> np.ndarray:
  """
  Interpolate outliers in an otherwise linear sequence.
  
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
  """
  Recursively interpolate outliers in sensor data.

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

def interpolate_skipped(
  diffs: np.ndarray,
  threshold: float = 0.3
) -> np.ndarray:
  """
  Corrects a series of time diffs (e.g., due to skipped beats) by detecting diffs 
  that deviate more than a given threshold from the median and then replacing them 
  using linear interpolation.
  
  Args:
    diffs: Array of beat-to-beat differences.
    threshold: Maximum allowed relative deviation from the median (e.g., 0.25 means 25%).
  Returns:
    np.ndarray: Corrected RR intervals.
  """
  median_diff = np.median(diffs)
  # Identify outliers
  outliers = np.abs(diffs - median_diff) > threshold * median_diff
  if not np.any(outliers):
    # No outliers, do nothing
    return diffs

  logging.debug(f"Interpolating {np.sum(outliers)} outlier events")
  idxs = np.arange(len(diffs))
  good_idxs = idxs[~outliers]
  good_diffs = diffs[~outliers]
  
  # Check if there are any good indices for interpolation
  if good_idxs.size == 0:
    logging.error("No good diff values found for interpolation, returning original diffs")
    return diffs

  corrected_diffs = diffs.copy()
  corrected_diffs[outliers] = np.interp(idxs[outliers], good_idxs, good_diffs)
  return corrected_diffs
