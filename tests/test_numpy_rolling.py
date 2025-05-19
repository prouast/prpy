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

from prpy.numpy.rolling import rolling_calc, rolling_calc_ragged

import numpy as np
import pytest

def test_rolling_calc_mean_no_transform():
  x = np.arange(10.0)
  res = rolling_calc(
    x,
    calc_fn=lambda v: np.mean(v, axis=-1),
    min_window_size=3,
    max_window_size=3,
  )
  assert np.isnan(res[:2]).all()
  assert pytest.approx(res[2]) == 1.0
  assert res.shape == x.shape

def test_rolling_calc_mean_with_transform():
  x = np.linspace(0, 2 * np.pi, 9)
  transform = lambda v: v - np.mean(v, axis=-1, keepdims=True)
  res = rolling_calc(
    x,
    calc_fn=lambda v: np.mean(np.abs(v), axis=-1),
    min_window_size=3,
    max_window_size=3,
    transform_fn=transform,
  )
  assert not np.isnan(res[2])
  assert res.shape == x.shape

def test_rolling_calc_short_input_only_pad():
  x = np.array([1.0, 2.0])
  res = rolling_calc(
    x,
    calc_fn=lambda v: np.mean(v, axis=-1),
    min_window_size=3,
    max_window_size=3,
  )
  assert np.isnan(res).all()

def test_rolling_calc_ragged_basic_count():
  """Counts of detections per ragged window are splashed onto each sample."""
  ts = np.array([0.3, 0.7, 1.2, 1.7, 2.1, 2.6])
  out = rolling_calc_ragged(
    ts,
    calc_fn=lambda seg: seg.size,
    min_window_size=0.5,
    max_window_size=1.0
  )
  exp = np.array([np.nan, 2., 3., 2., 3., 3.])
  np.testing.assert_array_equal(out, exp)

def test_rolling_calc_ragged_rate_from_mean_interval():
  """Median/mean-diff example – just sanity-check a few slots."""
  ts  = np.array([0.3, 0.7, 1.2, 1.7, 2.1, 2.6, 3.1])
  out = rolling_calc_ragged(
    ts,
    calc_fn=lambda seg: np.mean(np.diff(seg)),
    min_window_size=1.0,
    max_window_size=1.0,
  )
  exp = np.array([np.nan, np.nan, 0.45, 0.5, 0.45, 0.45, 0.5])
  np.testing.assert_allclose(out, exp)
