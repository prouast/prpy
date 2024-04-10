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

import numpy as np
import pytest

from prpy.numpy.metric import mag2db, mae, mse, rmse, cor, snr

@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 5)])
def test_mag2db(shape):
  out = mag2db(np.zeros(shape=shape))
  assert out.shape == shape

@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 5)])
def test_mae(shape):
  out = mae(np.zeros(shape=shape), np.zeros(shape=shape), axis=-1)
  assert out.shape == shape[:-1]

@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 5)])
def test_mse(shape):
  out = mse(np.zeros(shape=shape), np.zeros(shape=shape), axis=-1)
  assert out.shape == shape[:-1]

@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 5)])
def test_rmse(shape):
  out = rmse(np.zeros(shape=shape), np.zeros(shape=shape), axis=-1)
  assert out.shape == shape[:-1]

@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 5)])
def test_cor(shape):
  y_true = np.random.uniform(size=shape)
  y_pred = np.random.uniform(size=shape)
  out = cor(y_true, y_pred, axis=-1)
  assert out.shape == shape[:-1]
  if len(shape) == 1:
    np.testing.assert_allclose(out, np.corrcoef(y_true, y_pred)[0,1])

@pytest.mark.parametrize("shape", [(6,), (10, 6)])
def test_snr(shape):
  f_true = np.random.uniform(size=shape[:-1])
  y_pred = np.random.uniform(size=shape)
  out = snr(f_true, y_pred, f_s=5., f_res=.1)
  assert out.shape == shape[:-1]
