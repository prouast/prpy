# Copyright (c) 2024 Philipp Rouast
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

from prpy.numpy.signal import div0, normalize, standardize, moving_average, moving_average_size_for_response, moving_std, detrend
from prpy.numpy.signal import estimate_freq_fft, estimate_freq_peak, estimate_freq_periodogram
from prpy.numpy.signal import interpolate_vals, interpolate_cubic_spline, interpolate_linear_sequence_outliers, interpolate_data_outliers
from prpy.numpy.signal import _component_periodicity, select_most_periodic

import numpy as np
import pytest

@pytest.mark.parametrize("fill", [np.nan, np.inf, 0])
def test_div0(fill):
  # Check division by array including 0 with broadcasting
  a = np.array([[0, 1, 2], [3, 4, 5]])
  b = np.array([1, 0, 2])
  a_copy = a.copy()
  b_copy = b.copy()
  np.testing.assert_allclose(
    div0(a=a, b=b, fill=fill),
    np.array([[0, fill, 1], [3, fill, 2.5]]))
  # No side effects
  np.testing.assert_equal(a, a_copy)
  np.testing.assert_equal(b, b_copy)

def test_normalize():
  # Check with axis=-1
  np.testing.assert_allclose(
    normalize(x=np.array([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=-1),
    np.array([[0.2, 1.2, -2.8, 2.2, -0.8], [-1, 4, -4, 1, 0]]))
  # Check with axis=0
  np.testing.assert_allclose(
    normalize(x=np.array([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=0),
    np.array([[0., -2., 0., 0., -1.], [0., 2., 0., 0., 1.]]))
  # Check with axis=(0,1)
  x = np.array([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]])
  x_copy = x.copy()
  np.testing.assert_allclose(
    normalize(x=x, axis=(0,1)),
    np.array([[-.4, .6, -3.4, 1.6, -1.4], [-.4, 4.6, -3.4, 1.6, .6]]))
  # No side effects
  np.testing.assert_equal(x, x_copy)

def test_standardize():
  # Check with axis=-1
  np.testing.assert_allclose(
    standardize(x=np.array([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=-1),
    np.array([[.116247639, .697485834, -1.627466946, 1.278724029, -.464990556],
              [-.383482495, 1.533929979, -1.533929979, .383482495, 0.]]))
  # Check with axis=0
  np.testing.assert_allclose(
    standardize(x=np.array([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=0),
    np.array([[0., -1., 0., 0., -1.], [0., 1., 0., 0., 1.]]))
  # Check with axis=(0,1)
  np.testing.assert_allclose(
    standardize(x=np.array([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=(0,1)),
    np.array([[-.174740811, .262111217, -1.485296896, .698963245, -.61159284],
              [-.174740811, 2.00951933, -1.485296896, .698963245, .262111217]]))
  # No side effect
  x = np.array([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]])
  x_copy = x.copy()
  standardize(x)
  np.testing.assert_equal(x, x_copy)

def test_moving_average():
  # Check with axis=1
  np.testing.assert_allclose(
    moving_average(x=np.array([[0., .4, 1.2, 2., .2],
                               [0., .1, .5, .3, -.1],
                               [.3, 0., .3, -.2, -.6],
                               [.2, -.1, -1.2, -.4, .2]]),
                   axis=-1, size=3, pad_method='reflect', scale=False),
    np.array([[.133333333, .533333333, 1.2, 1.133333333, 0.8],
              [.033333333, .2, .3, .233333333, 0.033333333],
              [.2, .2, .033333333, -.166666667, -.466666667],
              [.1, -.366666667, -.566666667, -.466666667, 0.]]))
  # Check with axis=0
  np.testing.assert_allclose(
    moving_average(x=np.array([[0., .4, 1.2, 2., .2],
                               [0., .1, .5, .3, -.1],
                               [.3, 0., .3, -.2, -.6],
                               [.2, -.1, -1.2, -.4, .2]]),
                   axis=0, size=3, pad_method='reflect', scale=False),
    np.array([[0., .3, .966666667, 1.433333333, 0.1],
              [.1, .166666667, .666666667, .7, -.166666667],
              [.166666667, 0., -.133333333, -.1, -.166666667],
              [.233333334, -.066666667, -.7, -.333333333, -.066666667]]))
  # Check with small numbers and scale
  factor = 1000000.
  x = np.array([[0., .4, 1.2, 2., .2],
                [0., .1, .5, .3, -.1],
                [.3, 0., .3, -.2, -.6],
                [.2, -.1, -1.2, -.4, .2]])/factor
  x_copy = x.copy()
  np.testing.assert_allclose(
    moving_average(x=x,
                   axis=-1, size=3, pad_method='reflect', scale=True),
    np.array([[.133333333, .533333333, 1.2, 1.133333333, 0.8],
              [.033333333, .2, .3, .233333333, 0.033333333],
              [.2, .2, .033333333, -.166666667, -.466666667],
              [.1, -.366666667, -.566666667, -.466666667, 0.]])/factor)
  # No side effects
  np.testing.assert_equal(x, x_copy)

@pytest.mark.parametrize("cutoff_freq", [0.01, 0.1, 1, 10])
def test_moving_average_size_for_response(cutoff_freq):
  # Check that result >= 1
  assert moving_average_size_for_response(sampling_freq=1, cutoff_freq=cutoff_freq) >= 1

def test_moving_std():
  # Check a default use case
  x = np.array([0., .4, 1.2, 2., 2.1, 1.7, 11.4, 4.5, 1.9, 7.6, 6.3, 6.5])
  x_copy = x.copy()
  np.testing.assert_allclose(
    moving_std(x=x,
               size=5, overlap=3, fill_method='mean'),
    np.array([.83809307, .83809307, 2.35538328, 2.35538328, 2.79779145, 3.77764063, 3.57559311, 3.42705292, np.nan, np.nan, np.nan, np.nan]))
  # No side effects
  np.testing.assert_equal(x, x_copy)

def test_detrend():
  # Check a default use case with axis=-1
  z = np.array([[0., .4, 1.2, 2., .2],
                [0., .1, .5, .3, -.1],
                [.3, 0., .3, -.2, -.6],
                [.2, -.1, -1.2, -.4, .2]])
  z_copy = z.copy()
  np.testing.assert_allclose(
    detrend(z=z, Lambda=3, axis=-1),
    np.array([[-.29326743, -.18156859, .36271552, .99234445, -.88022395],
              [-.13389946, -.06970109, .309375, .12595109, -.23172554],
              [-.04370549, -.16474419, .31907328, .03090798, -.14153158],
              [.33796383, .15424475, -.86702586, -.08053786, .45535513]]),
    rtol=1e-6)
  # No side effects
  np.testing.assert_equal(z, z_copy)
  # Check a default use case with axis=0
  np.testing.assert_allclose(
    detrend(z=np.array([[0., .4, 1.2, 2., .2],
                        [0., .1, .5, .3, -.1],
                        [.3, 0., .3, -.2, -.6],
                        [.2, -.1, -1.2, -.4, .2]]), Lambda=3, axis=0),
    np.array([[.01093117, .05725853, -0.1004627, .39976865, .18635049],
              [-0.08016194, -.07703875, -.07755928, -.48877964, -.03799884],
              [.12753036, -.01769809, .45650665, -.22174667, -.48305379],
              [-.0582996, .03747831, -.27848467, .31075766, .33470214]]),
    rtol=1e-6)

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
    estimate_freq_peak(x=y, f_s=len(x), f_range=(max(freq-2,1),freq+2)),
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

def test_interpolate_cubic_spline():
  # Check a default use case
  x = np.array([0, 2, 3, 4, 7, 8, 9])
  y = np.array([[0.1, 0.4, 0.3, 0.1, 0.2, 0.25, 0.4],
                [0.1, 0.4, 0.3, 0.1, 0.2, 0.25, 0.4]])
  xs = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
  x_copy = x.copy()
  y_copy = y.copy()
  xs_copy = xs.copy()
  np.testing.assert_allclose(
    interpolate_cubic_spline(x=x, y=y, xs=xs, axis=-1),
    np.array([[.23781146, .38768688, .37072951, .19610327, .05045604, .06995432, .16060431, .22408638, .30091362, .57043189],
              [.23781146, .38768688, .37072951, .19610327, .05045604, .06995432, .16060431, .22408638, .30091362, .57043189]]))
  # No side effects
  np.testing.assert_equal(x, x_copy)
  np.testing.assert_equal(y, y_copy)
  np.testing.assert_equal(xs, xs_copy)

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
