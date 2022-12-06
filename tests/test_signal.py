###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import sys
sys.path.append('../propy')

from propy.signal import div0, normalize, standardize, moving_average

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
    moving_average(x=np.array([[0., .4, 1.2, 2., .2], [0., .1, .5, .3, -.1], [.3, 0., .3, -.2, -.6], [.2, -.1, -1.2, -.4, .2]]),
                   axis=-1, size=3, pad_method='reflect', scale=False),
    np.array([[.133333333, .533333333, 1.2, 1.133333333, 0.8],
              [.033333333, .2, .3, .233333333, 0.033333333],
              [.2, .2, .033333333, -.166666667, -.466666667],
              [.1, -.366666667, -.566666667, -.466666667, 0.]]))
  # Check with axis=0
  np.testing.assert_allclose(
    moving_average(x=np.array([[0., .4, 1.2, 2., .2], [0., .1, .5, .3, -.1], [.3, 0., .3, -.2, -.6], [.2, -.1, -1.2, -.4, .2]]),
                   axis=0, size=3, pad_method='reflect', scale=False),
    np.array([[0., .3, .966666667, 1.433333333, 0.1],
              [.1, .166666667, .666666667, .7, -.166666667],
              [.166666667, 0., -.133333333, -.1, -.166666667],
              [.233333334, -.066666667, -.7, -.333333333, -.066666667]]))
  # Check with small numbers and scale
  factor = 1000000.
  np.testing.assert_allclose(
    moving_average(x=np.array([[0., .4, 1.2, 2., .2], [0., .1, .5, .3, -.1], [.3, 0., .3, -.2, -.6], [.2, -.1, -1.2, -.4, .2]])/factor,
                   axis=-1, size=3, pad_method='reflect', scale=True),
    np.array([[.133333333, .533333333, 1.2, 1.133333333, 0.8],
              [.033333333, .2, .3, .233333333, 0.033333333],
              [.2, .2, .033333333, -.166666667, -.466666667],
              [.1, -.366666667, -.566666667, -.466666667, 0.]])/factor)

def 
  