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
from typing import Tuple, Optional

from prpy.constants import SECONDS_PER_MINUTE
from prpy.numpy.freq import estimate_freq
from prpy.numpy.interp import interpolate_skipped
from prpy.numpy.rolling import rolling_calc

BP_SYS_MIN = 40      # mmHg
BP_SYS_MAX = 240     # mmHg
BP_DIA_MIN = 20      # mmHg
BP_DIA_MAX = 160     # mmHg
HR_MIN = 40          # 1/min
HR_MAX = 240         # 1/min
HRV_SDNN_MIN = 1     # ms
HRV_SDNN_MAX = 200   # ms
PTT_MIN = 100        # ms
PTT_MAX = 400        # ms
IDX_MIN = 0.         
IDX_MAX = 1.
PWV_MIN = 0.2        # cm/ms
PWV_MAX = 0.6        # cm/ms
RR_MIN = 1           # 1/min
RR_MAX = 60          # 1/min
SPO2_MIN = 70        # %
SPO2_MAX = 100       # %

class EScope(IntEnum):
  """How the metric is computed along the signal timeline."""
  GLOBAL = 0
  ROLLING = 1

class EMethod(IntEnum):
  """Signal-analysis technique used to extract the rate/period."""
  PEAK = 0
  PERIODOGRAM = 1

def estimate_rate_per_minute_from_signal(
    signal: np.ndarray,
    f_s: float,
    f_range: Tuple[int, int],
    *,
    scope: EScope = EScope.GLOBAL,
    method: EMethod = EMethod.PERIODOGRAM,
    axis: int = -1,
    confidence: np.ndarray | None = None,
    window_size: int | None = None,
    overlap: int | None = None,
    interp_skipped: bool = False,
    **kw
  ) -> np.ndarray:
  # TODO: Integrate confidence
  """
  Estimate rate per minute from raw signal sampled at constant `f_s`.
  
  Args:
    signal: The raw sensor signal. Shape (n,) or (n_signals, n)
    f_s: Sampling frequency [Hz].
    f_range: Tuple (min, max) of plausible frequency range [Hz].
    scope: GLOBAL or ROLLING.
    method: PEAK, PERIODOGRAM, or FFT. 
    axis: Time axis of `signal`.
    confidence: Optional per-sample confidence mask. Same shape as `signal`.
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
                        pad_val=np.nan) * SECONDS_PER_MINUTE

def estimate_rate_per_minute_from_detections(
    det_idxs: np.ndarray,
    *,
    f_s: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    scope: EScope = EScope.GLOBAL,
    confidence: np.ndarray | None = None,
    window_size: int | None = None,
    overlap: int | None = None,
    interp_skipped: bool = False
  ) -> np.ndarray:
  # TODO: Integrate confidence
  """
  Estimate rate per minute from detection indices `det_idxs`.
  
  Args:
    det_idxs: The detection indices. Shape (n_dets,)
    f_s: The sampling rate.
    t: The timestamps. Shape (n,)
    scope: GLOBAL or ROLLING.
    confidence: Optional per-sample confidence mask. Same shape as `signal`.
    window_size: Window size in number of detections required for scope.ROLLING
    overlap: Overlap in number of detections for scope.ROLLING.
    interp_skipped: Insert interpolated detection for presumably skipped beats.
    **kw: Extra args forwarded to the underlying frequency estimator.
  Returns:
    The estimated rate.
      - For Scope.GLOBAL: Shape ()
      - For Scope.ROLLING: Same shape as `det_idxs`
  """
  if t is None and f_s is None: raise ValueError("Provide either `t` or `f_s`.")
  if scope is EScope.ROLLING:
    if window_size is None:
      raise ValueError("`window_size` is required when scope==Scope.ROLLING")
    if overlap is None: overlap = window_size - 1
  def _rate_from_dets(dets: np.ndarray) -> float:
    """Convert a 1-D array of detection indices to rate [1/min]"""
    if dets.size < 2: return np.nan
    det_t = (t[dets] if t is not None else dets / f_s).astype(float)
    diffs = np.diff(det_t)
    if interp_skipped:
      diffs = interpolate_skipped(diffs, threshold=0.3)
    if diffs.size == 0 or np.any(diffs == 0):
      return np.nan
    return 1 / np.median(diffs)
  def _window_calc(dets_view: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(_rate_from_dets, 1, dets_view)
  if scope == EScope.GLOBAL:
    return _rate_from_dets(det_idxs) * SECONDS_PER_MINUTE
  else:
    return rolling_calc(x=det_idxs,
                        calc_fn=_window_calc,
                        min_window_size=window_size,
                        max_window_size=window_size,
                        overlap=overlap,
                        pad_val=np.nan) * SECONDS_PER_MINUTE

def estimate_hr_from_signal(
    signal: np.ndarray,
    f_s: float,
    *,
    scope: EScope = EScope.GLOBAL,
    method: EMethod = EMethod.PERIODOGRAM,
    axis: int = -1,
    confidence: np.ndarray | None = None,
    window_size: int | None = None,
    overlap: int | None = None,
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
  return estimate_rate_per_minute_from_signal(
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
    confidence: np.ndarray | None = None,
    window_size: int | None = None,
    overlap: int | None = None,
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
  return estimate_rate_per_minute_from_signal(
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
