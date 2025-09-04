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

from dataclasses import dataclass
import logging
import numpy as np
from scipy import signal
from typing import Callable, Tuple, Optional, Union

from prpy.numpy.freq import estimate_freq_periodogram
from prpy.numpy.interp import interpolate_data_outliers
from prpy.numpy.rolling import rolling_calc

@dataclass
class PeakDetectDebug:
  t: np.ndarray
  vals_raw: np.ndarray
  vals_trans: np.ndarray
  freqs: np.ndarray
  det_idxs: np.ndarray
  det_idxs_raw: Optional[np.ndarray]
  det_t_diff: np.ndarray
  det_t_diff_imp: np.ndarray
  min_dist_t: float
  min_width_t: Optional[float]
  period_rel_tol: Tuple[float, float]

def detect_valid_peaks(
    vals: np.ndarray,
    *,
    f_s: Optional[float] = None,
    height: float = 0.,
    prominence: Optional[Tuple[float, float]] = None,
    period_rel_tol: Tuple[float, float] = (0.4, 0.8),
    window_size: int,
    overlap: Optional[int] = None,
    min_det_for_valid_seq: int = 1,
    t: Optional[np.ndarray] = None,
    width: Optional[Union[float, Tuple[float, float]]] = None,
    fft_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    f_range: Optional[Tuple[float, float]] = None,
    f_res: Optional[float] = None,
    vals_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    interp_vals_outliers_z: Optional[float] = None,
    interp_freqs_outliers_z: Optional[float] = None,
    refine: Optional[str] = None,
    refine_dist: float = 0.1,
    return_debug: bool = False,
  ) -> tuple:
  """
  Detect sequences of valid peaks in a periodical signal (e.g., ECG)
  
  Args:
    vals: The signal values to search for peaks (n,)
    f_s: Sampling frequency in Hz. Required if `t` not given.
    height: Required height of peaks
    prominence: Required (min, max) prominence of peaks
    period_rel_tol: (shorter, longer)
      - Allowed *relative* deviation of each peak-to-peak interval from the reference interval:
        valid if  (1-shorter)·T_ref  <  T  <  (1+longer)·T_ref
    window_size: The size of the reference frequency calculation window in number of data points
    overlap: The overlap of consecutive reference frequency calculation windows in number of data points
    min_det_for_valid_seq: Minimum consecutive valid detections to keep a sequence.
    t: The timestamps of the signal values (n,). If omitted, derived from `f_s`.
    width: Required width of peaks in seconds (either scalar as minimum or tuple for (min, max))
    fft_fn: Optional callable to be applied to vals before fft
    f_range: The frequency range to be considered (may be None)
    f_res: The frequency resolution (may be None)
    vals_fn: Optional transform applied before peak detection
    interp_vals_outliers_z, interp_freqs_outliers_z: Z-score thresholds for outlier interpolation.
    refine: Secondary peak-refinement strategy (None, 'raw_peak', 'raw_foot')
    refine_dist: Max admissible shift (seconds) during refinement.
    return_debug: If True, also return a `PeakDetectDebug` object.
  Returns:
    Tuple of
     - valid_det_idxs_list: List of lists of indices representing sequences of valid peaks
     - valid: Boolean array indicating which parts of the signal had valid peak detections (n,)
     - debug: PeakDetectDebug
  """
  # Basic checks
  if t is None and f_s is None: raise ValueError("Provide either `t` or `f_s`.")
  assert isinstance(vals, np.ndarray)
  size = vals.shape[0]
  if t is None:
    t = np.arange(size) / f_s
  else:
    assert t.shape[0] == size, "`t` must match `vals` length"
    if f_s is None:
      f_s = size / (t[-1] - t[0])
  vals_raw = vals.copy()
  period_rel_tol_shorter, period_rel_tol_longer = period_rel_tol
  # Sliding reference freq
  window_size = min(window_size, size)
  if overlap is None: overlap = window_size // 2
  freqs = rolling_calc(
    x=vals,
    calc_fn=lambda x: estimate_freq_periodogram(
      x=x,
      f_s=f_s,
      f_range=f_range,
      f_res=f_res,
      axis=1,
      keepdims=True
    ),
    min_window_size=window_size,
    max_window_size=window_size,
    overlap=overlap,
    transform_fn=fft_fn,
    fill_method='start',
    rolling_pad_mode='reflect'
  )
  assert freqs.shape == vals.shape
  # Preprocess
  if interp_vals_outliers_z is not None:
    vals = interpolate_data_outliers(vals, z_score=interp_vals_outliers_z)
  if vals_fn is not None:
    vals = vals_fn(vals)
  if interp_freqs_outliers_z is not None:
    freqs = interpolate_data_outliers(freqs, z_score=interp_freqs_outliers_z)
  # Peak detection
  if width is not None:
    width_s = width
    width = width * f_s if isinstance(width, float) else tuple(w * f_s if w is not None else None for w in width)
  else:
    width_s = None  
  min_dist_samples = max(1/np.quantile(freqs, 0.9) * f_s * (1 - period_rel_tol_shorter), 0)
  raw_det_idxs, _ = signal.find_peaks(
    vals,
    height=height,
    distance=min_dist_samples,
    width=width,
    prominence=prominence
  )
  if raw_det_idxs.size == 0:
    logging.warning("No peaks found - maybe tweak `vals_fn` or thresholds.")
    return ([], np.zeros(size)) if not return_debug else ([], np.zeros(size), None)
  if raw_det_idxs.size == 1:
    logging.warning("Only a single peak - periodicity checks will pass trivially.")
    valid = np.ones(size)
    return ([raw_det_idxs.tolist()], valid) if not return_debug \
        else ([raw_det_idxs.tolist()], valid, None)
  # Optional refinement
  det_idxs = raw_det_idxs
  if refine is not None:
    window_samples = int(refine_dist * f_s)
    if refine == 'raw_peak':
      det_idxs = _refine_raw_peak(vals=vals_raw,
                                  det_idxs=raw_det_idxs,
                                  window_samples=window_samples)
    elif refine == 'raw_foot':
      det_idxs = _refine_raw_foot(vals=vals_raw,
                                  det_idxs=raw_det_idxs,
                                  window_samples=window_samples)
    else:
      raise ValueError(f"Unknown refine mode '{refine}'.")
  # Validity test
  det_t_diff = np.diff(t[det_idxs])
  det_t_diff = np.concatenate([[det_t_diff[0]], det_t_diff])
  det_t_diff_imp = 1/freqs[det_idxs]
  # Detections are valid if diffs are close enough to implied diffs
  det_valid = np.logical_and(det_t_diff < det_t_diff_imp * (1 + period_rel_tol_longer),
                             det_t_diff > det_t_diff_imp * (1 - period_rel_tol_shorter))
  # Determine sequences of consecutive valid detections
  det_valid_seq_start_idx = np.where(np.logical_and(np.concatenate([[True], np.diff(det_valid)]), det_valid))[0]
  det_valid_seq_end_idx = np.where(np.logical_and(np.concatenate([np.diff(det_valid), [True]]), det_valid))[0]
  valid_det_idxs_list = [det_idxs[start:end+1].tolist() for start, end in zip(det_valid_seq_start_idx, det_valid_seq_end_idx) if end-start >= min_det_for_valid_seq]
  # Determine valid column
  valid_idxs_list = [list(range(det_idxs[start], det_idxs[end])) for start, end in zip(det_valid_seq_start_idx, det_valid_seq_end_idx)]
  valid_idxs = [item for sublist in valid_idxs_list for item in sublist]
  valid = np.zeros(size)
  valid[valid_idxs] = 1
  if not return_debug:
    return valid_det_idxs_list, valid
  debug = PeakDetectDebug(
    t=t,
    vals_raw=vals_raw,
    vals_trans=vals,
    freqs=freqs,
    det_idxs=det_idxs,
    det_idxs_raw=raw_det_idxs,
    det_t_diff=det_t_diff,
    det_t_diff_imp=det_t_diff_imp,
    min_dist_t=min_dist_samples / f_s,
    min_width_t=width_s[0] if isinstance(width_s, tuple) else width_s,
    period_rel_tol=period_rel_tol,
  )
  return valid_det_idxs_list, valid, debug

def _refine_raw_peak(
    vals: np.ndarray,
    det_idxs: np.ndarray,
    window_samples: int
  ) -> list:
  """
  Refine peak detections by searching for maximum within local window.

  Args:
    vals: Signal values (n,)
    det_idxs: Detected peak indices
    window_samples: Number of samples defining the local search window around each detection
  Returns:
    Lost of lists of refined peak indices
  """
  half_w = window_samples // 2
  offsets = np.arange(-half_w, half_w)
  
  # build a (N, window) index matrix around each detection
  idx_matrix = det_idxs[:, None] + offsets[None, :]
  idx_clipped = np.clip(idx_matrix, 0, vals.shape[0] - 1)
  
  # extract and mask out‑of‑bounds
  mask = (idx_matrix < 0) | (idx_matrix >= vals.shape[0])
  windows = vals[idx_clipped]
  windows = np.where(mask, -np.inf, windows)
  
  # find local max per row
  local_off = np.argmax(windows, axis=1)
  refined = idx_clipped[np.arange(det_idxs.size), local_off]
  
  return refined

def _refine_raw_foot(
    vals: np.ndarray,
    det_idxs: np.ndarray,
    window_samples: int
  ) -> list:
  """
  Refine peak detections by searching for minimum within local window.

  Args:
    vals: Signal values (n,)
    det_idxs: Detected peak indices
    window_samples: Number of samples defining the local search window around each detection
  Returns:
    Lost of lists of refined foot indices
  """  
  half_w = window_samples // 2
  offsets = np.arange(-half_w, half_w)
  
  # build a (N, window) index matrix around each detection
  idx_matrix = det_idxs[:, None] + offsets[None, :]
  idx_clipped = np.clip(idx_matrix, 0, vals.shape[0] - 1)
  
  # extract and mask out‑of‑bounds
  mask = (idx_matrix < 0) | (idx_matrix >= vals.shape[0])
  windows = vals[idx_clipped]
  windows = np.where(mask, np.inf, windows)
  
  # find local max per row
  local_off = np.argmin(windows, axis=1)
  refined = idx_clipped[np.arange(det_idxs.size), local_off]

  return refined
