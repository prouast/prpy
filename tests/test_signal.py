###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import sys
sys.path.append('../propy')

from propy.signal import div0, normalize, standardize, moving_average, moving_average_size_for_response, moving_std, detrend
from propy.signal import estimate_freq_fft, estimate_freq_peak, estimate_freq_periodogram

import numpy as np
import pytest

@pytest.mark.parametrize("fill", [np.nan, np.inf, 0])
def test_div0(fill):
  # Check division by array including 0 with broadcasting
  np.testing.assert_allclose(
    div0(a=np.array([[0, 1, 2], [3, 4, 5]]), b=np.array([1, 0, 2]), fill=fill),
    np.array([[0, fill, 1], [3, fill, 2.5]]))

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
  np.testing.assert_allclose(
    normalize(x=np.array([[0., 1., -3., 2., -1.], [0., 5., -3., 2., 1.]]), axis=(0,1)),
    np.array([[-.4, .6, -3.4, 1.6, -1.4], [-.4, 4.6, -3.4, 1.6, .6]]))

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
  np.testing.assert_allclose(
    moving_average(x=np.array([[0., .4, 1.2, 2., .2],
                               [0., .1, .5, .3, -.1],
                               [.3, 0., .3, -.2, -.6],
                               [.2, -.1, -1.2, -.4, .2]])/factor,
                   axis=-1, size=3, pad_method='reflect', scale=True),
    np.array([[.133333333, .533333333, 1.2, 1.133333333, 0.8],
              [.033333333, .2, .3, .233333333, 0.033333333],
              [.2, .2, .033333333, -.166666667, -.466666667],
              [.1, -.366666667, -.566666667, -.466666667, 0.]])/factor)

@pytest.mark.parametrize("cutoff_freq", [0.01, 0.1, 1, 10])
def test_moving_average_size_for_response(cutoff_freq):
  # Check that result >= 1
  assert moving_average_size_for_response(sampling_freq=1, cutoff_freq=cutoff_freq) >= 1

def test_moving_std():
  # Check a default use case
  np.testing.assert_allclose(
    moving_std(x=np.array([0., .4, 1.2, 2., 2.1, 1.7, 11.4, 4.5, 1.9, 7.6, 6.3, 6.5]),
               size=5, overlap=3, fill_method='mean'),
    np.array([.83809307, .83809307, 2.35538328, 2.35538328, 2.79779145, 3.77764063, 3.57559311, 3.42705292, np.nan, np.nan, np.nan, np.nan]))

def test_detrend():
  # Check a default use case with axis=-1
  np.testing.assert_allclose(
    detrend(z=np.array([[0., .4, 1.2, 2., .2],
                        [0., .1, .5, .3, -.1],
                        [.3, 0., .3, -.2, -.6],
                        [.2, -.1, -1.2, -.4, .2]]), Lambda=3, axis=-1),
    np.array([[-.29326743, -.18156859, .36271552, .99234445, -.88022395],
              [-.13389946, -.06970109, .309375, .12595109, -.23172554],
              [-.04370549, -.16474419, .31907328, .03090798, -.14153158],
              [.33796383, .15424475, -.86702586, -.08053786, .45535513]]),
    rtol=1e-6)
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
  y_ = 100 * np.sin(x) + np.random.normal(scale=8, size=x.size)
  y = np.stack([y_, y_], axis=0)
  # Check a default use case with axis=-1
  np.testing.assert_allclose(
    estimate_freq_fft(x=y, f_s=len(x), f_range=(max(freq-2,1),freq+2)),
    np.array([freq, freq]))

@pytest.mark.parametrize("num", [100, 500, 1000])
@pytest.mark.parametrize("freq", [5., 10., 20.])
def test_estimate_freq_peak(num, freq):
  # Test data
  x = np.linspace(0, freq * 2 * np.pi, num=num)
  np.random.seed(0)
  y_ = 100 * np.sin(x) + np.random.normal(scale=8, size=x.size)
  y = np.stack([y_, y_], axis=0)
  # Check a default use case with axis=-1
  np.testing.assert_allclose(
    estimate_freq_peak(x=y, f_s=len(x), f_range=(max(freq-2,1),freq+2)),
    np.array([freq, freq]),
    rtol=0.2)

@pytest.mark.parametrize("num", [100, 500, 1000])
@pytest.mark.parametrize("freq", [2.35, 4.89, 13.55])
def test_estimate_freq_periodogram(num, freq):
  # Test data
  x = np.linspace(0, freq * 2 * np.pi, num=num)
  np.random.seed(0)
  y_ = 100 * np.sin(x) + np.random.normal(scale=8, size=x.size)
  y = np.stack([y_, y_], axis=0)
  # Check a default use case with axis=-1
  np.testing.assert_allclose(
    estimate_freq_periodogram(x=y, f_s=len(x), f_range=(max(freq-2,1),freq+2), f_res=0.05),
    np.array([freq, freq]),
    rtol=0.01)
