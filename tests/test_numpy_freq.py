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

from prpy.numpy.freq import estimate_freq_fft, estimate_freq_peak, estimate_freq_periodogram
from prpy.numpy.freq import _component_periodicity, select_most_periodic

import numpy as np
import pytest

@pytest.mark.parametrize("num", [100, 500, 1000])
@pytest.mark.parametrize("freq", [1., 5., 10., 20.])
def test_estimate_freq_fft(num, freq):
  # Test data
  x = np.linspace(0, freq * 2 * np.pi, num=num)
  np.random.seed(0)
  y_ = 100 * np.sin(x) + np.random.normal(scale=8, size=num)
  y = np.stack([y_, y_], axis=0)
  y_copy = y.copy()
  # Check a default use case with axis=-1
  np.testing.assert_allclose(
    estimate_freq_fft(x=y, f_s=len(x), f_range=(max(freq-2,1),freq+2)),
    np.array([freq, freq]))
  # No side effects
  np.testing.assert_equal(y, y_copy)

@pytest.mark.parametrize("num", [100, 500, 1000])
@pytest.mark.parametrize("freq", [5., 10., 20.])
def test_estimate_freq_peak(num, freq):
  # Test data
  x = np.linspace(0, freq * 2 * np.pi, num=num)
  np.random.seed(0)
  y_ = 100 * np.sin(x) + np.random.normal(scale=8, size=num)
  y = np.stack([y_, y_], axis=0)
  y_copy = y.copy()
  # Check a default use case with axis=-1
  np.testing.assert_allclose(
    estimate_freq_peak(x=y,
                       f_s=len(x),
                       f_range=(max(freq-2,1),freq+2),
                       window_size=num),
    np.array([freq, freq]),
    rtol=0.2)
  # No side effects
  np.testing.assert_equal(y, y_copy)

@pytest.mark.parametrize("num", [100, 500, 1000])
@pytest.mark.parametrize("freq", [2.35, 4.89, 13.55])
def test_estimate_freq_periodogram(num, freq):
  # Test data
  x = np.linspace(0, freq * 2 * np.pi, num=num)
  np.random.seed(0)
  y_ = 100 * np.sin(x) + np.random.normal(scale=8, size=num)
  y = np.stack([y_, y_], axis=0)
  y_copy = y.copy()
  # Check a default use case with axis=-1
  np.testing.assert_allclose(
    estimate_freq_periodogram(x=y, f_s=len(x), f_range=(max(freq-2,1),freq+2), f_res=0.05),
    np.array([freq, freq]),
    rtol=0.01)
  # No side effects
  np.testing.assert_equal(y, y_copy)

def test_component_periodicity():
  # Test data
  x = np.stack([np.linspace(0, 1.3 * 2 * np.pi, num=300),
                np.linspace(0, 3.6 * 2 * np.pi, num=300),
                np.linspace(0, 7.1 * 2 * np.pi, num=300)], axis=0)
  np.random.seed(0)
  y = 100 * np.sin(x) + np.random.normal(scale=20, size=300)
  y_copy = y.copy()
  # Check default use case
  np.testing.assert_allclose(
    _component_periodicity(x=y),
    np.array([0.700568859, 0.46774879, 0.875988998]))
  # No side effects
  np.testing.assert_equal(y, y_copy)

def test_select_most_periodic():
  # Test data
  x = np.linspace(0, 3.6 * 2 * np.pi, num=300)
  np.random.seed(0)
  y = np.stack([100 * np.sin(x) + np.random.normal(scale=50, size=300),
                100 * np.sin(x) + np.random.normal(scale=10, size=300), # least noise
                100 * np.sin(x) + np.random.normal(scale=100, size=300)], axis=0)
  y_copy = y.copy()
  # Check default use case
  np.testing.assert_allclose(
    select_most_periodic(x=y),
    y[1])
  # No side effects
  np.testing.assert_equal(y, y_copy)
