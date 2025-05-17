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

from prpy.numpy.rolling import rolling_calc

import numpy as np
import pytest

def test_rolling_mean_no_transform():
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

def test_rolling_mean_with_transform():
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

def test_short_input_only_pad():
  x = np.array([1.0, 2.0])
  res = rolling_calc(
    x,
    calc_fn=lambda v: np.mean(v, axis=-1),
    min_window_size=3,
    max_window_size=3,
  )
  assert np.isnan(res).all()
