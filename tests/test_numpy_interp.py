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

from prpy.numpy.interp import interpolate_vals, interpolate_linear_sequence_outliers
from prpy.numpy.interp import interpolate_data_outliers, interpolate_filtered, interpolate_skipped

import numpy as np
import pytest

def test_interpolate_vals():
  # Check a default use case
  x = np.array([np.nan, 0., 0.2, np.nan, np.nan, 0.4, 1.2, 3.1])
  x_copy = x.copy()
  np.testing.assert_allclose(
    interpolate_vals(x=x),
    np.array([0., 0., 0.2, 0.266666667, 0.333333333, 0.4, 1.2, 3.1]))
  # No side effects
  np.testing.assert_equal(x, x_copy)
  # Check with all nan
  np.testing.assert_allclose(
    interpolate_vals(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])),
    np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]))

@pytest.mark.sklearn
def test_interpolate_linear_sequence_outliers():
  # Check an easy case
  t = np.array([1, 4, 14, 12, 16])
  t_copy = t.copy()
  np.testing.assert_allclose(
    interpolate_linear_sequence_outliers(t=t),
    np.array([1, 4, 8, 12, 16]))
  # No side effects
  np.testing.assert_equal(t, t_copy)
  # More complicated
  np.testing.assert_allclose(
    interpolate_linear_sequence_outliers(
      np.array([1, 4, 7, 12, 12, 18, 21, 30, 28, 33, 37])),
    np.array([1, 4, 7, 12, 15, 18, 21, 24.5, 28, 33, 37]))
  # First element outlier
  np.testing.assert_allclose(
    interpolate_linear_sequence_outliers(
      np.array([80, 4, 7, 12, 12, 18, 21, 30, 28, 33, 37])),
    np.array([1, 4, 7, 12, 15, 18, 21, 24.5, 28, 33, 37]))
  # Last element outlier
  np.testing.assert_allclose(
    interpolate_linear_sequence_outliers(
      np.array([1, 4, 7, 12, 12, 18, 21, 30, 28, 33, 90])),
    np.array([1, 4, 7, 12, 15, 18, 21, 24.5, 28, 33, 38]))
  # Double outlier
  np.testing.assert_allclose(
    interpolate_linear_sequence_outliers(
      np.array([1, 4, 7, 12, 79, 83, 21, 30, 28, 33, 38])),
    np.array([1, 4, 7, 12, 15, 18, 21, 30, 31.5, 33, 38]))
  # Longer sequence with massive outliers and duplicate value
  np.testing.assert_allclose(
    interpolate_linear_sequence_outliers(
      np.array([1, 4, 6, 10, 50, 16, 20, 24, 27, 30, 32, 35, -300, 40, 43, 46, 50, 50, 53, 56, 58, 61])),
    np.array([1, 4, 6, 10, 13, 16, 20, 24, 27, 30, 32, 35, 37.5, 40, 43, 46, 50, 51.5, 53, 56, 58, 61]))

def test_interpolate_data_outliers():
  # Check an easy case
  x = np.array([0.9, 1.6, 1.1, 1.3, 1.6, 1000.0, 0.8, 1.0, 1.3, 0.9, 0.8, 1.2])
  x_copy = x.copy()
  np.testing.assert_allclose(
    interpolate_data_outliers(x=x, z_score=3.),
    np.array([0.9, 1.6, 1.1, 1.3, 1.6, 1.2, 0.8, 1.0, 1.3, 0.9, 0.8, 1.2]))
  # No side effects
  np.testing.assert_equal(x, x_copy)
  # Two outliers, next to each other in middle
  np.testing.assert_allclose(
    interpolate_data_outliers(
      np.array([0.9, 1.6, 1.1, 1.3, 1.6, 1.2, 1000., 1000., 1.3, 0.9, 0.8, 1.2]), z_score=2.),
    np.array([0.9, 1.6, 1.1, 1.3, 1.6, 1.2, 1.23333, 1.266667, 1.3, 0.9, 0.8, 1.2]),
    atol=1e-4, rtol=1e-4)
  # Two outliers, one at edge
  np.testing.assert_allclose(
    interpolate_data_outliers(
      np.array([1000., 1.6, 1.1, 1.3, 1.6, 1000.0, 0.8, 1.0, 1.3, 0.9, 0.8, 1.2]), z_score=2.),
    np.array([1.163636, 1.6, 1.1, 1.3, 1.6, 1.2, 0.8, 1.0, 1.3, 0.9, 0.8, 1.2]),
    atol=1e-4, rtol=1e-4)
  # Two outliers, both at edge
  np.testing.assert_allclose(
    interpolate_data_outliers(
      np.array([0.9, 1.6, 1.1, 1.3, 1.6, 1.2, 0.8, 1.0, 1.3, 0.9, 1000., 1000.]), z_score=2.),
    np.array([0.9, 1.6, 1.1, 1.3, 1.6, 1.2, 0.8, 1.0, 1.3, 0.9, 0.5, 1.109091]),
    atol=1e-4, rtol=1e-4)

@pytest.mark.parametrize(
  "t_in,s_in,t_out,band,order,s_out_exp,fill_nan",
  [
    (
      np.array([0, 2, 3, 4, 7, 8, 9], dtype=float),
      np.array([0.1, 0.4, 0.3, 0.1, np.nan, 0.25, 0.4]),
      np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
      None,
      4,
      np.array([0.235938, 0.382812, 0.366667, 0.183333, 0.105729, 0.142187, 0.191146, 0.228437, 0.306719]),
      True
    ),
    (
      np.array([0, 2, 3, 4, 7, 8, 9], dtype=float),
      np.array([0.1, 0.4, 0.3, 0.1, np.nan, np.nan, 0.4]),
      np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]),
      None,
      4,
      np.array([0.235938, 0.382812, 0.366667, 0.183333, 0.1003, 0.1081, 0.1375, np.nan, 0.3187]),
      False
    ),
    (
      np.array([0., 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03, 0.032, 0.034, 0.036, 0.038]),
      np.array([0.03259748, -0.03539362, 0.10271604, 0.16905992, np.nan, np.nan, np.nan, 0.49898612, 0.42037613, 0.59396127, 0.68605126, 0.6030388, 0.5608828, 0.79987835, 0.86046953, 0.78139013, 0.7121775, 0.93807187, 0.90422569, 1.03567749]),
      np.array([0., 0.01, 0.02, 0.03]),
      (0.5, 40),
      2,
      np.array([0.032597, np.nan, 0.686051, 0.78139]),
      False
    ),
    (
      np.array([0., 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03, 0.032, 0.034, 0.036, 0.038]),
      np.array([0.03259748, -0.03539362, 0.10271604, 0.16905992, 0.06804494, 0.23287238, np.nan, 0.49898612, 0.42037613, 0.59396127, 0.68605126, 0.6030388, 0.5608828, 0.79987835, 0.86046953, 0.78139013, 0.7121775, 0.93807187, 0.90422569, 1.03567749]),
      np.array([0., 0.01, 0.02, 0.03]),
      (0.5, 40),
      2,
      np.array([-0.636099, -1.021032, -0.622056, -0.610314]),
      True
    ),
    (
      np.array([0., 1., 2.], dtype=float),
      np.array([10., 20., 30.]),
      np.array([-1., 0.5, 1.5, 2.5]),
      None,
      4,
      np.array([np.nan, 15., 25., np.nan]),
      True
    )
  ]
)
@pytest.mark.parametrize("extradim", [False, True])
def test_interpolate_filtered(t_in, s_in, t_out, band, order, s_out_exp, fill_nan, extradim):
  if extradim:
    s_in = np.asarray([s_in, s_in])
    s_out_exp = np.asarray([s_out_exp, s_out_exp])
  # Immutable originals
  t_in_copy = t_in.copy()
  s_in_copy = s_in.copy()
  t_out_copy = t_out.copy()
  # Run and compare
  s_out = interpolate_filtered(t_in=t_in, s_in=s_in, t_out=t_out, band=band, order=order, fill_nan=fill_nan, axis=-1)
  np.testing.assert_allclose(s_out, s_out_exp, atol=1e-6)
  # Test no side effects
  np.testing.assert_equal(t_in, t_in_copy)
  np.testing.assert_equal(s_in, s_in_copy)
  np.testing.assert_equal(t_out, t_out_copy)

@pytest.mark.parametrize("rr_intervals,threshold,expected", [
  (np.array([1, 1, 1, 1, 1]), 0.25, np.array([1, 1, 1, 1, 1])),
  (np.array([1, 1, 2, 1, 1]), 0.25, np.array([1, 1, 1, 1, 1])),
  (np.array([1, .8, 2, .9, 1]), 0.25, np.array([1, .8, .85, .9, 1])),
  (np.array([2, 1, 1, 1, 2]), 0.25, np.array([1, 1, 1, 1, 1])),
  (np.array([1, 1, 2, 1, 1]), 1.5, np.array([1, 1, 2, 1, 1])),
])
def test_interpolate_skipped(rr_intervals, threshold, expected):
  corrected = interpolate_skipped(rr_intervals, threshold)
  np.testing.assert_allclose(corrected, expected, atol=1e-2)
