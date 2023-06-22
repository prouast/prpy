###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, January 2023                #
###############################################################################

import sys
sys.path.append('../propy')

import numpy as np
import pytest

from propy.numpy.metric import mag2db, mae, mse, rmse, cor, snr

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

@pytest.mark.parametrize("shape", [(6,), (2, 6)])
def test_snr(shape):
  f_true = np.random.uniform(size=shape[:-1])
  y_pred = np.random.uniform(size=shape)
  out = snr(f_true, y_pred, f_s=5., f_res=.1)
  assert out.shape == shape[:-1]

