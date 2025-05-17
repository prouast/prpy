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

from prpy.numpy.core import div0, normalize, standardize

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
