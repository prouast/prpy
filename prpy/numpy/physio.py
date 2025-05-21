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

from enum import IntEnum
import numpy as np
from typing import Tuple, Optional, List, Callable, Union

from prpy.constants import SECONDS_PER_MINUTE, MILLIS_PER_SECOND
from prpy.numpy.detect import detect_valid_peaks
from prpy.numpy.filters import moving_average_size_for_response
from prpy.numpy.freq import estimate_freq
from prpy.numpy.interp import interpolate_skipped
from prpy.numpy.rolling import rolling_calc, rolling_calc_ragged

BP_SYS_MIN = 40           # mmHg
BP_SYS_MAX = 240          # mmHg
BP_DIA_MIN = 20           # mmHg
BP_DIA_MAX = 160          # mmHg
HR_MIN = 40               # 1/min
HR_MAX = 240              # 1/min
HRV_SDNN_MIN = 1          # ms
HRV_SDNN_MAX = 200        # ms
PTT_MIN = 100             # ms
PTT_MAX = 400             # ms
IDX_MIN = 0.              # unitless
IDX_MAX = 1.              # unitless
PWV_MIN = 0.2             # cm/ms
PWV_MAX = 0.6             # cm/ms
RR_MIN = 1                # 1/min
RR_MAX = 60               # 1/min
SPO2_MIN = 70             # %
SPO2_MAX = 100            # %

CALC_HR_MIN_T = 5         # seconds
CALC_HR_MAX_T = 10        # seconds
CALC_HRV_SDNN_MIN_T = 10  # seconds
CALC_HRV_SDNN_MAX_T = 60  # seconds
CALC_RR_MIN_T = 10        # seconds
CALC_RR_MAX_T = 30        # seconds

class EScope(IntEnum):
  """How the metric is computed along the signal timeline."""
  GLOBAL = 0
  ROLLING = 1

class EMethod(IntEnum):
  """Signal-analysis technique used to extract the rate/period."""
  PEAK = 0
  PERIODOGRAM = 1

class EWindowUnit(IntEnum):
  """Unit for rolling window."""
  DETECTIONS = 0
  SECONDS = 1

def estimate_rate_from_signal(
    signal: np.ndarray,
    f_s: float,
    f_range: Tuple[int, int],
    *,
    scope: EScope = EScope.GLOBAL,
    method: EMethod = EMethod.PERIODOGRAM,
    axis: int = -1,
    window_size: Optional[int] = None,
    overlap: Optional[int] = None,
    interp_skipped: bool = False,
    pad_val: float = np.nan,
    **kw
  ) -> np.ndarray:
  """
  Estimate rate per minute from raw signal sampled at constant `f_s`.
  
  Args:
    signal: The raw sensor signal. Shape (n,) or (n_signals, n)
    f_s: Sampling frequency [Hz].
    f_range: Tuple (min, max) of plausible frequency range [Hz].
    scope: GLOBAL for scalar rate or ROLLING for rate trace shape (n,).
    method: PEAK, PERIODOGRAM, or FFT. 
    axis: Time axis of `signal`.
    window_size: Window size in number of signal data points required for scope.ROLLING
    overlap: Overlap in number of signal data points for scope.ROLLING.
    interp_skipped: Insert interpolated detection for presumably skipped beats.
    **kw: Extra args forwarded to the underlying frequency estimator.
  Returns:
    The estimated rate.
      - For Scope.GLOBAL: Shape (n_signals,)
      - For Scope.ROLLING: Same shape as `signal`
  """
  # Validate special kwargs
  if scope is EScope.ROLLING:
    if window_size is None:
      raise ValueError("`window_size` is required when scope==Scope.ROLLING")
    if overlap is None: overlap = window_size // 2
  # Pick estimator
  estimate_fn = lambda x: estimate_freq(
    x=x,
    f_s=f_s,
    f_range=(f_range[0], f_range[1]),
    f_res=0.5/SECONDS_PER_MINUTE,
    method=method.name.lower(),
    interp_skipped=interp_skipped,
    axis=axis,
    **kw,
    **({'window_size': window_size} if window_size is not None else {}),
    **({'overlap': overlap} if overlap is not None else {})
  )
  if scope == EScope.GLOBAL:
    return estimate_fn(signal) * SECONDS_PER_MINUTE
  else:
    return rolling_calc(x=signal,
                        calc_fn=estimate_fn,
                        min_window_size=window_size,
                        max_window_size=window_size,
                        overlap=overlap,
                        pad_val=pad_val) * SECONDS_PER_MINUTE

def _calc_from_detections(
    det_idxs: np.ndarray,
    calc_fn_from_dets: Callable,
    calc_fn_from_ts: Callable,
    *,
    f_s: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    scope: EScope = EScope.GLOBAL,
    min_window_size: Optional[float] = None,
    max_window_size: Optional[float] = None,
    overlap: Optional[int] = None,
    window_unit: EWindowUnit = EWindowUnit.DETECTIONS,
    pad_val: float = np.nan,
  ) -> np.ndarray:
  """
  Apply reducer `calc_fn_from_dets` / `calc_fn_from_ts` to detections `det_idxs`.
  
  Args:
    det_idxs: The detection indices. Shape (n_dets,)
    calc_fn_from_dets: The function to be applied to dets. Must accept 1-D input and reduce to 0-D.
    calc_fn_from_ts: The function to be applied to ts. Must accept 1-D input and reduce to 0-D.
    f_s: The sampling rate. Required when `t` is not given.
    t: The timestamps of the original signal. Required for scope.ROLLING. Shape (n,)
    scope: GLOBAL for scalar rate or ROLLING for rate trace shape (n,).
    min_window_size: Minimum window size for scope.ROLLING in the `window_unit`.
    max_window_size: Maximum window size for scope.ROLLING in the `window_unit`.
    overlap: Overlap for scope.ROLLING in the `window_unit`.
    pad_val: Value for padding.
  Returns:
    The results.
      - For Scope.GLOBAL: Shape ()
      - For Scope.ROLLING: Shape (n,)
  """
  det_idxs = np.asarray(det_idxs, dtype=float) if t is None else np.asarray(det_idxs, dtype=int)
  if det_idxs.ndim != 1: raise ValueError("`det_idxs` must be 1-D")
  if scope is EScope.GLOBAL:
    if t is None and f_s is None: raise ValueError("Provide either `t` or `f_s`.")
  else:
    if t is None: raise ValueError("Provide `t`.")
  if scope is EScope.GLOBAL:
    return calc_fn_from_dets(det_idxs)
  if min_window_size is None or max_window_size is None:
    raise ValueError("`min_window_size` and `max_window_size` are required when scope==Scope.ROLLING")
  if overlap is None:
    overlap = min_window_size - 1 if window_unit is EWindowUnit.DETECTIONS else None
  if overlap is not None and overlap > min_window_size:
    raise ValueError("`overlap` must be <= `min_window_size`")
  if window_unit is EWindowUnit.DETECTIONS:
    # Count-based window
    result_per_det = rolling_calc(
      x=det_idxs,
      calc_fn=lambda v: np.apply_along_axis(calc_fn_from_dets, 1, v),
      min_window_size=int(min_window_size),
      max_window_size=int(max_window_size),
      overlap=int(overlap),
      pad_val=pad_val
    )
  else:
    # Duration-based window
    det_t = t[det_idxs].astype(float)
    result_per_det = rolling_calc_ragged(
      x=det_t,
      calc_fn=calc_fn_from_ts,
      min_window_size=min_window_size,
      max_window_size=max_window_size,
      pad_val=pad_val,
    )
  # Up-sample “step” trace
  t_full = t.astype(float)
  det_t = t[det_idxs].astype(float)
  result_full = np.full_like(t_full, pad_val, dtype=float)
  if det_t.size:
    idx = np.searchsorted(det_t, t_full, side='right') - 1
    valid = idx >= 0
    result_full[valid] = result_per_det[idx[valid]]
    if det_t.size >= 2:
      cutoff = det_t[-1] + (det_t[-1] - det_t[-2])
      result_full[t_full >= cutoff] = pad_val
  return result_full

def _calc_from_detection_sequences(
    seqs: List[np.ndarray],
    calc_fn_from_dets: Callable,
    calc_fn_from_ts: Callable,
    *,
    f_s: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    scope: EScope = EScope.GLOBAL,
    pad_val: float = np.nan,
    **kw,
  ):
  """
  Apply reducer `calc_fn_from_dets` / `calc_fn_from_ts` to one OR many sequences
  of detections and merge the results.
  
  Args:
    seqs: List of np.ndarray sequences of detections.
    calc_fn_from_dets: The function to be applied to dets. Must accept 1-D input and reduce to 0-D.
    calc_fn_from_ts: The function to be applied to ts. Must accept 1-D input and reduce to 0-D.
    f_s: The sampling rate. Required when `t` is not given.
    t: The timestamps of the original signal. Required for scope.ROLLING. Shape (n,)
    scope: GLOBAL for scalar rate or ROLLING for rate trace shape (n,).
    pad_val: Value for padding.
    **kw: Further keywords passed on to `_calc_from_detections`.
  Returns:
    The results.
      - For Scope.GLOBAL: Shape ()
      - For Scope.ROLLING: Shape (n,)
  """
  if not seqs:
    raise ValueError("`seqs` must contain at least one sequence")
  if scope is EScope.GLOBAL:
    if t is None and f_s is None: raise ValueError("Provide either `t` or `f_s`.")
  else:
    if t is None: raise ValueError("Provide `t`.")
  if scope is EScope.GLOBAL and t is None:
    def _stitch(seqs: List[np.ndarray]) -> np.ndarray:
      """Merge runs, dropping the first index of each later run and
      left-shifting the remainder so that diffs are preserved."""
      _seqs = [np.asarray(s, dtype=int) for s in seqs if s.size]
      _seqs.sort(key=lambda a: a[0])
      merged = _seqs[0].copy()
      for s in _seqs[1:]:
        if s.size < 2:
          continue
        gap = s[0] - merged[-1]
        if gap <= 0:
          raise ValueError("Sequences overlap or are not strictly ascending")
        merged = np.concatenate([merged, s[1:] - gap])
      return merged
    merged_idxs = _stitch(seqs)
    return _calc_from_detections(
      det_idxs=merged_idxs,
      calc_fn_from_dets=calc_fn_from_dets,
      calc_fn_from_ts=calc_fn_from_ts,
      f_s=f_s,
      t=None,
      scope=EScope.GLOBAL,
      pad_val=pad_val,
      **kw,
    )
  if scope is EScope.GLOBAL:
    results, weights = [], []
    for s in seqs:
      r = _calc_from_detections(
        det_idxs=s,
        calc_fn_from_dets=calc_fn_from_dets,
        calc_fn_from_ts=calc_fn_from_ts,
        f_s=f_s,
        t=t,
        scope=EScope.GLOBAL,
        pad_val=pad_val,
        **kw
      )
      results.append(r)
      dur = (t[s[-1]] - t[s[0]]) if t is not None else (s[-1] - s[0]) / f_s
      weights.append(dur)
    results = np.array(results, float)
    weights = np.array(weights, float)
    return np.average(results, weights=weights)
  results_full = np.full(t.shape[0], pad_val)
  for s in seqs:
    r_seq = _calc_from_detections(
      det_idxs=s,
      calc_fn_from_dets=calc_fn_from_dets,
      calc_fn_from_ts=calc_fn_from_ts,
      f_s=f_s,
      t=t,
      scope=EScope.ROLLING,
      pad_val=pad_val,
      **kw
    )
    mask = ~np.isnan(r_seq)
    results_full[mask] = r_seq[mask]
  return results_full

def estimate_rate_from_detections(
    det_idxs: np.ndarray,
    *,
    f_s: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    scope: EScope = EScope.GLOBAL,
    min_window_size: Optional[float] = None,
    max_window_size: Optional[float] = None,
    overlap: Optional[int] = None,
    window_unit: EWindowUnit = EWindowUnit.DETECTIONS,
    interp_skipped: bool = False,
    pad_val: float = np.nan,
  ) -> np.ndarray:
  """
  Estimate rate per minute from detection indices `det_idxs`.
  
  Args:
    det_idxs: The detection indices. Shape (n_dets,)
    f_s: The sampling rate. Required when `t` is not given.
    t: The timestamps of the original signal. Required for scope.ROLLING. Shape (n,)
    scope: GLOBAL for scalar rate or ROLLING for rate trace shape (n,).
    min_window_size: Minimum window size for scope.ROLLING in the `window_unit`.
    max_window_size: Maximum window size for scope.ROLLING in the `window_unit`.
    overlap: Overlap for scope.ROLLING in the `window_unit`.
    window_unit: DETECTIONS or SECONDS.
    interp_skipped: Insert interpolated detection for presumably skipped beats.
    pad_val: Value for padding.
  Returns:
    The estimated rate.
      - For Scope.GLOBAL: Shape ()
      - For Scope.ROLLING: Shape (n,)
  """
  def _rate_from_ts(det_t: np.ndarray) -> float:
    """Convert a 1-D array of detection times to rate [1/min]"""
    if det_t.size < 2: return np.nan
    diffs = np.diff(det_t)
    if interp_skipped:
      diffs = interpolate_skipped(diffs, threshold=0.3)
    if diffs.size == 0 or np.any(diffs == 0):
      return np.nan
    return SECONDS_PER_MINUTE / np.median(diffs)
  def _rate_from_dets(dets: np.ndarray) -> float:
    """Convert a 1-D array of detection indices to rate [1/min]"""
    dets = np.asarray(dets)
    dets = dets[~np.isnan(dets)].astype(int, copy=False)
    if dets.size < 2: return np.nan
    det_t = (t[dets] if t is not None else dets / f_s).astype(float)
    return _rate_from_ts(det_t)
  return _calc_from_detections(
    det_idxs=det_idxs,
    calc_fn_from_dets=_rate_from_dets,
    calc_fn_from_ts=_rate_from_ts,
    f_s=f_s,
    t=t,
    scope=scope,
    min_window_size=min_window_size,
    max_window_size=max_window_size,
    overlap=overlap,
    window_unit=window_unit,
    pad_val=pad_val
  )

def estimate_rate_from_detection_sequences(
    seqs: List[np.ndarray],
    *,
    f_s: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    scope: EScope = EScope.GLOBAL,
    min_window_size: Optional[float] = None,
    max_window_size: Optional[float] = None,
    overlap: Optional[int] = None,
    window_unit: EWindowUnit = EWindowUnit.DETECTIONS,
    interp_skipped: bool = False,
    pad_val: float = np.nan,
  ) -> np.ndarray:
  """
  Estimate rate per minute from to one or many sequences of detections.
  
  Args:
    seqs: List of np.ndarray sequences of detections.
    f_s: The sampling rate. Required when `t` is not given.
    t: The timestamps of the original signal. Required for scope.ROLLING. Shape (n,)
    scope: GLOBAL for scalar rate or ROLLING for rate trace shape (n,).
    min_window_size: Minimum window size for scope.ROLLING in the `window_unit`.
    max_window_size: Maximum window size for scope.ROLLING in the `window_unit`.
    overlap: Overlap for scope.ROLLING in the `window_unit`.
    window_unit: DETECTIONS or SECONDS.
    interp_skipped: Insert interpolated detection for presumably skipped beats.
    pad_val: Value for padding.
  Returns:
    The results.
      - For Scope.GLOBAL: Shape ()
      - For Scope.ROLLING: Shape (n,)
  """
  def _rate_from_ts(det_t: np.ndarray) -> float:
    """Convert a 1-D array of detection times to rate [1/min]"""
    if det_t.size < 2: return np.nan
    diffs = np.diff(det_t)
    if interp_skipped:
      diffs = interpolate_skipped(diffs, threshold=0.3)
    if diffs.size == 0 or np.any(diffs == 0):
      return np.nan
    return SECONDS_PER_MINUTE / np.median(diffs)
  def _rate_from_dets(dets: np.ndarray) -> float:
    """Convert a 1-D array of detection indices to rate [1/min]"""
    dets = np.asarray(dets)
    dets = dets[~np.isnan(dets)].astype(int, copy=False)
    if dets.size < 2: return np.nan
    det_t = (t[dets] if t is not None else dets / f_s).astype(float)
    return _rate_from_ts(det_t)
  return _calc_from_detection_sequences(
    seqs=seqs,
    calc_fn_from_dets=_rate_from_dets,
    calc_fn_from_ts=_rate_from_ts,
    f_s=f_s,
    t=t,
    scope=scope,
    min_window_size=min_window_size,
    max_window_size=max_window_size,
    overlap=overlap,
    window_unit=window_unit,
    pad_val=pad_val
  )

def estimate_hr_from_signal(
    signal: np.ndarray,
    f_s: float,
    *,
    scope: EScope = EScope.GLOBAL,
    method: EMethod = EMethod.PERIODOGRAM,
    axis: int = -1,
    confidence: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    overlap: Optional[int] = None,
    interp_skipped: bool = False,
    **kw
  ) -> np.ndarray:
  """
  Estimate heart rate from raw signal sampled at constant `f_s`.
  
  Args:
    signal: The raw sensor signal. Shape (n,) or (n_signals, n)
    f_s: Sampling frequency [Hz].
    scope: GLOBAL or ROLLING.
    method: PEAK, PERIODOGRAM, or FFT. 
    axis: Time axis of `signal`.
    confidence: Optional per-sample confidence mask. Same shape as `signal`.
    window_size: Window size in number of signal data points required for scope.ROLLING
    overlap: Overlap in number of signal data points for scope.ROLLING.
    interp_skipped: Insert interpolated detection for presumably skipped beats
    **kw: Extra args forwarded to the underlying frequency estimator.
  Returns:
    The estimated heart rate.
      - For Scope.GLOBAL: Shape (n_signals,)
      - For Scope.ROLLING: Same shape as `signal`
  """
  return estimate_rate_from_signal(
    signal=signal,
    f_s=f_s,
    f_range=(HR_MIN/SECONDS_PER_MINUTE, HR_MAX/SECONDS_PER_MINUTE),
    scope=scope,
    method=method,
    axis=axis,
    confidence=confidence,
    window_size=window_size,
    overlap=overlap,
    interp_skipped=interp_skipped,
    **kw
  )

def estimate_rr_from_signal(
    signal: np.ndarray,
    f_s: float,
    *,
    scope: EScope = EScope.GLOBAL,
    method: EMethod = EMethod.PERIODOGRAM,
    axis: int = -1,
    confidence: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
    overlap: Optional[int] = None,
    interp_skipped: bool = False,
    **kw
  ) -> np.ndarray:
  """
  Estimate respiratory rate from raw signal sampled at constant `f_s`.
  
  Args:
    signal: The raw sensor signal. Shape (n,) or (n_signals, n)
    f_s: Sampling frequency [Hz].
    scope: GLOBAL or ROLLING.
    method: PEAK, PERIODOGRAM, or FFT. 
    axis: Time axis of `signal`.
    confidence: Optional per-sample confidence mask. Same shape as `signal`.
    window_size: Window size in number of signal data points required for scope.ROLLING
    overlap: Overlap in number of signal data points for scope.ROLLING.
    interp_skipped: Insert interpolated detection for presumably skipped breaths
    **kw: Extra args forwarded to the underlying frequency estimator.
  Returns:
    The estimated respiratory rate.
      - For Scope.GLOBAL: Shape (n_signals,)
      - For Scope.ROLLING: Same shape as `signal`
  """
  return estimate_rate_from_signal(
    signal=signal,
    f_s=f_s,
    f_range=(RR_MIN/SECONDS_PER_MINUTE, RR_MAX/SECONDS_PER_MINUTE),
    scope=scope,
    method=method,
    axis=axis,
    confidence=confidence,
    window_size=window_size,
    overlap=overlap,
    interp_skipped=interp_skipped,
    **kw
  )

def estimate_hrv_sdnn_from_signal(
    signal: np.ndarray,
    f_s: float,
    min_window_size: float,
    max_window_size: float,
    *,
    scope: EScope = EScope.GLOBAL,
    overlap: Optional[float] = None,
    confidence: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.6,
    window_unit: EWindowUnit = EWindowUnit.DETECTIONS,
    interp_skipped: bool = False,
    min_dets: int = 8,
    min_t: float = 10.,
    pad_val: float = np.nan,
    **kw
  ) -> Tuple[np.ndarray, float]:
  """
  Estimate HRV (SDNN) from raw signal sampled at constant `f_s`.
  
  Args:
    signal: The raw sensor signal. Shape (n,)
    f_s: Sampling frequency [Hz].
    f_range: Tuple (min, max) of plausible frequency range [Hz].
    scope: GLOBAL for scalar hrv or ROLLING for hrv trace shape (n,).
    min_window_size: Minimum window size for scope.ROLLING in the `window_unit`.
    max_window_size: Maximum window size for scope.ROLLING in the `window_unit`.
    overlap: Overlap for scope.ROLLING in the `window_unit`.
    confidence: Optional confidences for the raw sensor signal. Shape (n,)
    confidence_threshold: Confidence threshold above which to consider detected peaks
    window_unit: DETECTIONS or SECONDS.
    interp_skipped: Insert interpolated detection for presumably skipped beats.
    min_dets: Minimum number of valid dets required for calculation.
    min_t: Minimum duration of signal [seconds] required for calculation.
    **kw: Extra args forwarded to the peak detector.
  Returns:
    The estimated rate.
      - For Scope.GLOBAL: Shape ()
      - For Scope.ROLLING: Same shape as `signal`
  """
  if overlap is None and window_unit is EWindowUnit.SECONDS:
    overlap = max(min_window_size, max_window_size // 2)
  if overlap is None and window_unit is EWindowUnit.DETECTIONS:
    overlap = max(min_window_size, max_window_size - 1)
  # Detect peaks
  det_idxs, _ = detect_valid_peaks(
    vals=signal,
    f_s=f_s,
    f_range=(HR_MIN/SECONDS_PER_MINUTE, HR_MAX/SECONDS_PER_MINUTE),
    window_size=max_window_size,
    overlap=overlap,
    **kw
  )
  # Confidence
  def _conf_filter_and_split(seqs, conf, thr):
    seqs = [np.array(run, dtype=int) for run in seqs]
    return [sub for seq in seqs for sub in np.split(
        seq[conf[seq] >= thr],
        np.where(np.diff(np.where(conf[seq] >= thr)[0]) != 1)[0] + 1
      ) if sub.size
    ]
  def _lowest_kept_conf(seqs, conf, thr):
    if conf is None: return 0.0
    kept = np.concatenate(_conf_filter_and_split(seqs, conf, thr)) \
          if seqs else np.empty(0, int)
    return float(np.nanmin(conf[kept])) if kept.size else 0.0
  if confidence is not None:
    # Remove low-confidence indices
    det_idxs = _conf_filter_and_split(det_idxs, confidence, confidence_threshold)
    sdnn_conf = _lowest_kept_conf(det_idxs, confidence, confidence_threshold)
  if len(det_idxs) == 0:
     sdnn = np.nan if scope is EScope.GLOBAL else np.full(signal.shape, np.nan)
     return sdnn, 0.
  # Continue using the detections
  sdnn = estimate_hrv_sdnn_from_detection_sequences(
    seqs=det_idxs,
    f_s=f_s,
    scope=scope,
    min_window_size=min_window_size,
    max_window_size=max_window_size,
    overlap=overlap,
    window_unit=window_unit,
    interp_skipped=interp_skipped,
    min_dets=min_dets,
    min_t=min_t,
    pad_val=pad_val
  )
  return sdnn, sdnn_conf

def estimate_hrv_sdnn_from_detections(
    det_idxs: np.ndarray,
    *,
    f_s: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    scope: EScope = EScope.GLOBAL,
    min_window_size: Optional[float] = None,
    max_window_size: Optional[float] = None,
    overlap: Optional[int] = None,
    window_unit: EWindowUnit = EWindowUnit.DETECTIONS,
    interp_skipped: bool = False,
    min_dets: int = 8,
    min_t: float = 10.,
    correct_quantization_error: bool = True,
    pad_val: float = np.nan,
  ) -> np.ndarray:
  """
  Estimate HRV (SDNN) from detection indices `det_idxs`.
  
  Args:
    det_idxs: The detection indices. Shape (n_dets,)
    f_s: The sampling rate. Required when `t` is not given.
    t: The timestamps of the original signal. Required for scope.ROLLING. Shape (n,)
    scope: GLOBAL for scalar rate or ROLLING for rate trace shape (n,).
    min_window_size: Minimum window size for scope.ROLLING in the `window_unit`.
    max_window_size: Maximum window size for scope.ROLLING in the `window_unit`.
    overlap: Overlap for scope.ROLLING in the `window_unit`.
    window_unit: DETECTIONS or SECONDS.
    interp_skipped: Insert interpolated detection for presumably skipped beats.
    min_dets: Minimum number of valid dets required for calculation.
    min_t: Minimum duration of signal [seconds] required for calculation.
    pad_val: Value for padding.
  Returns:
    The results.
      - For Scope.GLOBAL: Shape ()
      - For Scope.ROLLING: Shape (n,)
  """
  if f_s is None: f_s = t.shape[0]/(t[-1]-t[0])
  def _rate_from_ts(det_t: np.ndarray) -> float:
    """Convert a 1-D array of detection times to hrv sdnn [ms]"""
    diffs = np.diff(det_t)
    if interp_skipped:
      diffs = interpolate_skipped(diffs, threshold=0.3)
    if diffs.size - 1 < min_dets or det_t[-1] - det_t[0] < min_t or np.any(diffs == 0):
      # Signal not sufficient for estimation
      return np.nan
    var_e = 1./(12*f_s**2) if correct_quantization_error else 0
    hrv_sdnn = np.sqrt(np.nanvar(diffs) - var_e) * MILLIS_PER_SECOND
    if hrv_sdnn > HRV_SDNN_MAX: return HRV_SDNN_MAX
    return hrv_sdnn
  def _rate_from_dets(dets: np.ndarray) -> float:
    """Convert a 1-D array of detection indices to hrv sdnn [ms]"""
    dets = np.asarray(dets)
    dets = dets[~np.isnan(dets)].astype(int, copy=False)
    det_t = (t[dets] if t is not None else dets / f_s).astype(float)
    return _rate_from_ts(det_t)
  return _calc_from_detections(
    det_idxs=det_idxs,
    calc_fn_from_dets=_rate_from_dets,
    calc_fn_from_ts=_rate_from_ts,
    f_s=f_s,
    t=t,
    scope=scope,
    min_window_size=min_window_size,
    max_window_size=max_window_size,
    overlap=overlap,
    window_unit=window_unit,
    pad_val=pad_val
  )

def estimate_hrv_sdnn_from_detection_sequences(
    seqs: List[np.ndarray],
    *,
    f_s: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    scope: EScope = EScope.GLOBAL,
    min_window_size: Optional[float] = None,
    max_window_size: Optional[float] = None,
    overlap: Optional[int] = None,
    window_unit: EWindowUnit = EWindowUnit.DETECTIONS,
    interp_skipped: bool = False,
    min_dets: int = 8,
    min_t: float = 10.,
    correct_quantization_error: bool = True,
    pad_val: float = np.nan,
  ) -> np.ndarray:
  """
  Estimate HRV (SDNN) from to one or many sequences of detections.
  
  Args:
    seqs: List of np.ndarray sequences of detections.
    f_s: The sampling rate. Required when `t` is not given.
    t: The timestamps of the original signal. Required for scope.ROLLING. Shape (n,)
    scope: GLOBAL for scalar rate or ROLLING for rate trace shape (n,).
    min_window_size: Minimum window size for scope.ROLLING in the `window_unit`.
    max_window_size: Maximum window size for scope.ROLLING in the `window_unit`.
    overlap: Overlap for scope.ROLLING in the `window_unit`.
    window_unit: DETECTIONS or SECONDS.
    interp_skipped: Insert interpolated detection for presumably skipped beats.
    min_dets: Minimum number of valid dets required for calculation.
    min_t: Minimum duration of signal [seconds] required for calculation.
    pad_val: Value for padding.
  Returns:
    The results.
      - For Scope.GLOBAL: Shape ()
      - For Scope.ROLLING: Shape (n,)
  """
  if f_s is None: f_s = t.shape[0]/(t[-1]-t[0])
  def _rate_from_ts(det_t: np.ndarray) -> float:
    """Convert a 1-D array of detection times to hrv sdnn [ms]"""
    diffs = np.diff(det_t)
    if interp_skipped:
      diffs = interpolate_skipped(diffs, threshold=0.3)
    if diffs.size + 1 < min_dets or det_t[-1] - det_t[0] < min_t or np.any(diffs == 0):
      # Signal not sufficient for estimation
      return np.nan
    var_e = 1./(12*f_s**2) if correct_quantization_error else 0
    hrv_sdnn = np.sqrt(np.nanvar(diffs) - var_e) * MILLIS_PER_SECOND
    if hrv_sdnn > HRV_SDNN_MAX: return HRV_SDNN_MAX
    return hrv_sdnn
  def _rate_from_dets(dets: np.ndarray) -> float:
    """Convert a 1-D array of detection indices to hrv sdnn [ms]"""
    dets = np.asarray(dets)
    dets = dets[~np.isnan(dets)].astype(int, copy=False)
    if dets.size < min_dets: return np.nan
    det_t = (t[dets] if t is not None else dets / f_s).astype(float)
    return _rate_from_ts(det_t)
  return _calc_from_detection_sequences(
    seqs=seqs,
    calc_fn_from_dets=_rate_from_dets,
    calc_fn_from_ts=_rate_from_ts,
    f_s=f_s,
    t=t,
    scope=scope,
    min_window_size=min_window_size,
    max_window_size=max_window_size,
    overlap=overlap,
    window_unit=window_unit,
    pad_val=pad_val
  )

def moving_average_size_for_hr_response(
    f_s: Union[float, int]
  ) -> int:
  """Get the moving average window size for a signal with HR information sampled at a given frequency
  
  Args:
    f_s: The sampling frequency
  Returns:
    The moving average size in number of signal vals
  """
  return moving_average_size_for_response(f_s, HR_MAX / SECONDS_PER_MINUTE)

def moving_average_size_for_rr_response(
    f_s: Union[float, int]
  ) -> int:
  """Get the moving average window size for a signal with RR information sampled at a given frequency
  
  Args:
    f_s: The sampling frequency
  Returns:
    The moving average size in number of signal vals
  """
  return moving_average_size_for_response(f_s, RR_MAX / SECONDS_PER_MINUTE)

def detrend_lambda_for_hr_response(
    f_s: Union[float, int]
  ) -> int:
  """Get the detrending lambda parameter for a signal with HR information sampled at a given frequency
  
  Args:
    f_s: The sampling frequency
  Returns:
    The lambda parameter
  """
  return int(0.1614*np.power(f_s, 1.9804))

def detrend_lambda_for_rr_response(
    f_s: Union[float, int]
  ) -> int:
  """Get the detrending lambda parameter for a signal with RR information sampled at a given frequency
  
  Args:
    f_s: The sampling frequency
  Returns:
    The lambda parameter
  """
  return int(4.4248*np.power(f_s, 2.1253))
