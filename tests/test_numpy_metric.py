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

import numpy as np
import pytest

from prpy.numpy.metric import _mag2db, mae, mse, rmse, cor, snr

def _make_pair(shape, seed=0, nan_ratio=0.25):
  rng = np.random.default_rng(seed)
  y_true = rng.normal(size=shape)
  y_pred = rng.normal(size=shape)
  nan_mask = rng.random(size=shape) < nan_ratio
  y_true[nan_mask] = np.nan
  y_pred[nan_mask & (rng.random(size=shape) < 0.5)] = np.nan
  return y_true, y_pred

nanmean_abs = lambda a, b, ax: np.nanmean(np.abs(a - b), axis=ax)
nanmean_sq = lambda a, b, ax: np.nanmean((a - b) ** 2, axis=ax)

def nan_corr_1d(a, b):
  m = np.isfinite(a) & np.isfinite(b)
  if m.sum() < 2: return np.nan
  a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
  return (a * b).mean() / (a.std() * b.std())

@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 5)])
def test_mag2db(shape):
  x = np.zeros(shape=shape)
  x_copy = x.copy()
  out = _mag2db(x)
  assert out.shape == shape
  np.testing.assert_equal(x, x_copy)

@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 5)])
def test_mae(shape):
  y_true = np.zeros(shape=shape)
  y_pred = np.zeros(shape=shape)
  y_true_copy = y_true.copy()
  y_pred_copy = y_pred.copy()
  out = mae(y_true=y_true, y_pred=y_pred, axis=-1)
  assert out.shape == shape[:-1]
  np.testing.assert_equal(y_true, y_true_copy)
  np.testing.assert_equal(y_pred, y_pred_copy)

@pytest.mark.parametrize("shape", [(4,), (2, 4), (2, 3, 4)])
def test_mae_ignore_nan(shape):
  y_true, y_pred = _make_pair(shape)
  y_true_copy, y_pred_copy = y_true.copy(), y_pred.copy()
  legacy = mae(y_true, y_pred, axis=-1)
  assert np.isnan(legacy).any()
  ref = nanmean_abs(y_true, y_pred, -1)
  np.testing.assert_allclose(mae(y_true, y_pred, ignore_nan=True), ref, equal_nan=True)
  np.testing.assert_equal(y_true, y_true_copy)
  np.testing.assert_equal(y_pred, y_pred_copy)

@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 5)])
def test_rmse(shape):
  y_true = np.zeros(shape=shape)
  y_pred = np.zeros(shape=shape)
  y_true_copy = y_true.copy()
  y_pred_copy = y_pred.copy()
  out = rmse(y_true=y_true, y_pred=y_pred, axis=-1)
  assert out.shape == shape[:-1]
  np.testing.assert_equal(y_true, y_true_copy)
  np.testing.assert_equal(y_pred, y_pred_copy)

@pytest.mark.parametrize("shape", [(4,), (2,4), (2,3,4)])
def test_mse_rmse_ignore_nan(shape):
  y_t, y_p = _make_pair(shape, seed=1)
  ref_mse  = nanmean_sq(y_t, y_p, -1)
  np.testing.assert_allclose(mse(y_t, y_p, ignore_nan=True),  ref_mse,  equal_nan=True)
  np.testing.assert_allclose(rmse(y_t, y_p, ignore_nan=True), np.sqrt(ref_mse), equal_nan=True)

@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 5)])
def test_cor(shape):
  y_true = np.random.uniform(size=shape)
  y_pred = np.random.uniform(size=shape)
  y_true_copy = y_true.copy()
  y_pred_copy = y_pred.copy()
  out = cor(y_true=y_true, y_pred=y_pred, axis=-1)
  assert out.shape == shape[:-1]
  if len(shape) == 1:
    np.testing.assert_allclose(out, np.corrcoef(y_true, y_pred)[0,1])
  np.testing.assert_equal(y_true, y_true_copy)
  np.testing.assert_equal(y_pred, y_pred_copy)

@pytest.mark.parametrize("shape", [(5,), (3,5), (2,3,5)])
def test_cor_ignore_nan(shape):
  y_t, y_p = _make_pair(shape, seed=2)
  ref = np.array([nan_corr_1d(a, b)
                  for a, b in zip(y_t.reshape(-1, shape[-1]),
                                  y_p.reshape(-1, shape[-1]))]).reshape(shape[:-1])
  np.testing.assert_allclose(cor(y_t, y_p, ignore_nan=True), ref, equal_nan=True)

@pytest.mark.parametrize("shape", [(6,), (10, 6)])
@pytest.mark.parametrize("f_s", [5., 5])
@pytest.mark.parametrize("tol", [.2, 1])
@pytest.mark.parametrize("f_min", [.5, 1])
@pytest.mark.parametrize("f_max", [4, 4.])
def test_snr(shape, f_s, tol, f_min, f_max):
  f_true = np.random.uniform(size=shape[:-1])
  y_pred = np.random.uniform(size=shape)
  f_true_copy = f_true.copy()
  y_pred_copy = y_pred.copy()
  out = snr(f_true=f_true, y_pred=y_pred, f_s=f_s, f_res=.1, tol=tol, f_min=f_min, f_max=f_max)
  assert out.shape == shape[:-1]
  np.testing.assert_equal(f_true, f_true_copy)
  np.testing.assert_equal(y_pred, y_pred_copy)
