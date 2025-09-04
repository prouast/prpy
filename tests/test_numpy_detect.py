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

import sys
sys.path.append('../prpy')

from prpy.numpy.core import standardize
from prpy.numpy.detect import detect_valid_peaks, _refine_raw_foot
from prpy.numpy.filters import moving_average

import numpy as np
import pytest
from typing import Optional

def _make_ecg_like(
    fs: int,
    duration: float,
    hr_bpm: float,
    *,
    missing_beat_at: Optional[int] = None,
    noise_std: float = 0.05,
  ) -> np.ndarray:
  n = int(duration * fs)
  t = np.arange(n) / fs
  rr = 60.0 / hr_bpm
  beats = np.arange(0, duration, rr)
  if missing_beat_at is not None and missing_beat_at < len(beats):
    beats = np.delete(beats, missing_beat_at)
  sig = np.zeros_like(t)
  for bt in beats:
    sig += np.exp(-(t - bt) ** 2 / (2 * (0.015 ** 2)))
  sig = sig - sig.mean() + np.random.normal(0, noise_std, n)
  sig = standardize(sig)
  return t, sig

def test_detect_valid_peaks_flat_signal():
  fs = 250
  t = np.arange(10 * fs) / fs
  vals = np.zeros_like(t)
  seqs, valid = detect_valid_peaks(
    vals,
    f_s=fs,
    height=0.1,
    prominence=(0.05, None),
    period_rel_tol=(0.3, 0.3),
    window_size=fs * 4,
    overlap=fs * 4 - 1,
  )
  assert seqs == []
  np.testing.assert_array_equal(valid, np.zeros_like(vals))

@pytest.mark.parametrize("missing_idx,exp_nseq", [(None, 1), (5, 2)])
def test_detect_valid_peaks_synthetic_ecg(missing_idx, exp_nseq):
  np.random.seed(0)
  fs = 250
  t, v = _make_ecg_like(fs, duration=20, hr_bpm=72, missing_beat_at=missing_idx)
  seqs, mask = detect_valid_peaks(
    vals=v,
    t=t,
    height=1.,
    prominence=(0.1, None),
    period_rel_tol=(0.25, 0.5),
    window_size=fs * 8,
    overlap=fs * 7,
    min_det_for_valid_seq=3,
    f_range=(0.5, 2),
    f_res=0.01,
    fft_fn=lambda x: moving_average(np.power(x, 2), size=101)
  )

  assert len(seqs) == exp_nseq

  # mask length must equal original length
  assert mask.shape == v.shape
  # mask non-zero only between first and last peak of each sequence
  for seq in seqs:
    assert mask[seq[0] : seq[-1]].all()

def test_detect_valid_peaks_outlier_interp():
  fs = 250
  t, v = _make_ecg_like(fs, duration=8, hr_bpm=60)
  v[fs * 3] += 20 # single large artefact

  seqs, _ = detect_valid_peaks(
    vals=v,
    f_s=fs,
    height=1.,
    prominence=(0.2, None),
    period_rel_tol=(0.25, 0.5),
    window_size=fs * 4,
    overlap=fs * 4 - 1,
    interp_vals_outliers_z=6,
    refine='raw_peak',
    f_range=(0.5, 2),
    f_res=0.01,
    fft_fn=lambda x: moving_average(np.power(x, 2), size=101)
  )
  # artefact should not split the main valid sequence
  assert len(seqs) == 1

def test_refine_raw_foot_edge_case():
  vals = np.array([10, 8, 2, 0, 5, 6, 7])
  det_idxs = np.array([1])
  window_samples = 6
  expected_refined_idx = 3
  refined_idxs = _refine_raw_foot(
    vals=vals,
    det_idxs=det_idxs,
    window_samples=window_samples
  )
  assert refined_idxs[0] == expected_refined_idx
