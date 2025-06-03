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

from prpy.numpy.metric import _mag2db, mae, mse, rmse, cor, snr, MAX_SHIFT_FIX
from prpy.numpy.metric import make_shift_pooled_metric, cor_shift, mae_shift, mse_shift

def _make_pair(shape: tuple[int, ...], *, seed=0, nan_ratio=0.25):
  """random normal + NaNs in identical positions for y_true / y_pred"""
  rng = np.random.default_rng(seed)
  y_t = rng.normal(size=shape)
  y_p = rng.normal(size=shape)
  m   = rng.random(size=shape) < nan_ratio
  y_t[m] = np.nan
  # 50 % of masked positions in y_pred also NaN
  y_p[m & (rng.random(size=shape) < .5)] = np.nan
  return y_t, y_p

def _build_sine_batch(batch, length, *, freq=1.0, fs=30.0,
                      shift=0, noise=0.0, seed=0):
  """Batch of pure sines with optional circular shift + additive noise."""
  t = np.arange(length) / fs
  base = np.sin(2 * np.pi * freq * t, dtype=np.float32)
  y_t  = np.repeat(base[None, :], batch, 0)
  y_p  = np.roll(y_t, -shift, axis=1).copy()      # «predictions»
  if noise > 0:
    rng = np.random.default_rng(seed)
    y_p += rng.normal(scale=noise, size=y_p.shape)
  return y_t, y_p

nanmean_abs = lambda a, b, ax: np.nanmean(np.abs(a - b), axis=ax)
nanmean_sq  = lambda a, b, ax: np.nanmean((a - b)**2, axis=ax)

def _nan_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
  m = np.isfinite(a) & np.isfinite(b)
  if m.sum() < 2:
    return np.nan
  a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
  return (a * b).mean() / (a.std() * b.std())

@pytest.mark.parametrize("shape", [(7,), (2, 5)])
def test_mag2db_returns_same_shape_and_no_mutation(shape):
  x   = np.ones(shape, dtype=float)
  x_c = x.copy()
  out = _mag2db(x)
  assert out.shape == shape
  np.testing.assert_equal(x, x_c)

@pytest.mark.parametrize("shape", [(4,), (3, 4)])
def test_mae_zero(shape):
  y = np.zeros(shape)
  np.testing.assert_allclose(mae(y, y), 0)

@pytest.mark.parametrize("shape", [(6,), (2, 6)])
def test_mae_ignore_nan_matches_numpy(shape):
  y_t, y_p = _make_pair(shape, seed=1)
  ref      = nanmean_abs(y_t, y_p, -1)
  np.testing.assert_allclose(mae(y_t, y_p, ignore_nan=True), ref, equal_nan=True)

@pytest.mark.parametrize("shape", [(5,), (2, 5)])
def test_mse_rmse_consistency(shape):
  y_t, y_p = _make_pair(shape, seed=2, nan_ratio=0)
  mse_val  = mse(y_t, y_p)
  rmse_val = rmse(y_t, y_p)
  np.testing.assert_allclose(rmse_val, np.sqrt(mse_val))

@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_rmse(shape):
  y_true = np.zeros(shape=shape)
  y_pred = np.zeros(shape=shape)
  y_true_copy = y_true.copy()
  y_pred_copy = y_pred.copy()
  out = rmse(y_true=y_true, y_pred=y_pred)
  assert out.shape == shape[:-1]
  np.testing.assert_equal(y_true, y_true_copy)
  np.testing.assert_equal(y_pred, y_pred_copy)

@pytest.mark.parametrize("shape", [(4,), (2,4)])
def test_mse_rmse_ignore_nan(shape):
  y_t, y_p = _make_pair(shape, seed=1)
  ref_mse  = nanmean_sq(y_t, y_p, -1)
  np.testing.assert_allclose(mse(y_t, y_p, ignore_nan=True),  ref_mse,  equal_nan=True)
  np.testing.assert_allclose(rmse(y_t, y_p, ignore_nan=True), np.sqrt(ref_mse), equal_nan=True)

@pytest.mark.parametrize("shape", [(5,), (3, 5)])
def test_cor_against_numpy(shape):
  y_t = np.random.default_rng(0).uniform(size=shape)
  y_p = np.random.default_rng(1).uniform(size=shape)
  out = cor(y_t, y_p)
  if len(shape) == 1:
    np.testing.assert_allclose(out, np.corrcoef(y_t, y_p)[0, 1])
  else:
    refs = [np.corrcoef(y_t[i], y_p[i])[0, 1] for i in range(shape[0])]
    np.testing.assert_allclose(out, refs)

@pytest.mark.parametrize("shape", [(6,), (2, 6)])
def test_cor_ignore_nan_matches_manual(shape):
  y_t, y_p = _make_pair(shape, seed=3)
  ref = np.asarray([_nan_corr_1d(a, b)
                    for a, b in zip(y_t.reshape(-1, shape[-1]),
                                    y_p.reshape(-1, shape[-1]))]
                  ).reshape(shape[:-1])
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

@pytest.mark.parametrize("shift", [0, 3, 7])
@pytest.mark.parametrize("metric_fn,maximise", [
  (mae, False),
  (mse, False),
  (cor, True),
])
def test_make_shift_pooled_metric_matches_bruteforce(metric_fn, maximise, shift):
  max_t   = 0.15
  metric  = make_shift_pooled_metric(metric_fn, maximise=maximise,
                                     max_shift_t=max_t)
  B, T    = 3, 64
  fs      = np.array([30, 25, 30], dtype=float)  # mix scalar/array later
  y_t, y_p = _build_sine_batch(B, T, shift=shift)
  out      = metric(y_t, y_p, fs, 0.7, 3.0)

  # brute-force best shift
  shifts   = np.arange(-MAX_SHIFT_FIX, MAX_SHIFT_FIX + 1)
  vals     = []
  for s in shifts:
    idx = np.arange(T)
    m   = idx >= s if s >= 0 else idx < (T + s)
    if not m.any():
      vals.append(np.full(B, np.inf))
      continue
    v = metric_fn(y_t[:, m], np.roll(y_p, s, axis=1)[:, m])
    vals.append(v)
  vals      = np.stack(vals)                       # (S,B)
  k_per     = np.minimum(np.floor(max_t * fs).astype(int), MAX_SHIFT_FIX)
  adm       = np.abs(shifts)[:, None] <= k_per[None, :]
  sign      = -1 if maximise else 1
  tmp   = np.where(adm, sign * vals, np.inf)
  best  = tmp.min(axis=0)
  expect = sign * best
  np.testing.assert_allclose(out, expect, atol=1e-6)

@pytest.mark.parametrize("fn", [mae, mse, rmse, cor])
def test_no_inplace_modification(fn):
  y_t, y_p = _make_pair((3, 8))
  y_t_c, y_p_c = y_t.copy(), y_p.copy()
  _ = fn(y_t, y_p)
  np.testing.assert_equal(y_t, y_t_c)
  np.testing.assert_equal(y_p, y_p_c)

def test_convenience_wrappers_basic():
  y_t, y_p = _build_sine_batch(1, 128, shift=6)
  fs       = 30
  assert mae_shift(y_t, y_p, fs, 0.7, 3.0) < mae(y_t, y_p)
  assert mse_shift(y_t, y_p, fs, 0.7, 3.0) < mse(y_t, y_p)
  assert cor_shift(y_t, y_p, fs, 0.7, 3.0) > cor(y_t, y_p)