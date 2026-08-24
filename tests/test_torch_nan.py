# Copyright (c) 2026 Philipp Rouast
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

"""Tests for prpy/torch/nan.py, using the exact same value/gradient scenarios
as tests/test_tensorflow.py's `test_reduce_nan{mean,sum}[_grad]` and
`test_nan_linear_combination` - these numbers were derived from (and already
verify) the TF implementation, so reusing them here is itself a cross-
framework parity check without needing TF importable in this env. The
gradient checks are the actual point: forward-only tests would miss the
"double where" NaN-gradient-poisoning bug class this module exists to avoid.
"""

import math

import numpy as np
import pytest
import torch

from prpy.torch.nan import reduce_nanmean, reduce_nansum
from prpy.torch.nan import ReduceNanMean, ReduceNanSum, NanLinearCombination


def assert_near_nan(x: torch.Tensor, y: torch.Tensor, tol=1e-6):
  # Broadcast first (TF's assert_equal/assert_near do this implicitly) -
  # some multi-axis-reduce scenarios below give an expected tensor with a
  # kept size-1 dim that only matches the actual output's shape after
  # broadcasting, not by exact shape equality.
  x, y = torch.broadcast_tensors(x, y)
  nan_mask_x = torch.isnan(x)
  nan_mask_y = torch.isnan(y)
  assert torch.equal(nan_mask_x, nan_mask_y), f"NaN masks differ: {nan_mask_x} vs {nan_mask_y}"
  torch.testing.assert_close(x[~nan_mask_x], y[~nan_mask_y], atol=tol, rtol=tol)


SCENARIOS_MEAN = [
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], 4. / 3., None),  # Reduce all
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[1., 2., math.nan], [1., 2., math.nan]], -1),  # Reduce one dim
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[1., 2., math.nan]], (0, 2)),  # Reduce multiple dims
  ([[math.nan, math.nan], [math.nan, math.nan], [math.nan, math.nan]], [[math.nan, math.nan, math.nan]], (0, 2)),
]


@pytest.mark.parametrize("scenario", SCENARIOS_MEAN)
def test_reduce_nanmean(scenario):
  x_list, y_list, dim = scenario
  x = torch.tensor([x_list, x_list])
  y = torch.tensor(y_list)
  assert_near_nan(reduce_nanmean(x, dim=dim), y)


SCENARIOS_MEAN_GRAD = [
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], 4. / 3.,
   [[1. / 6., 1. / 6.], [1. / 6., 0.], [0., 0.]], None),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[1., 2., math.nan], [1., 2., math.nan]],
   [[1. / 2., 1. / 2.], [1., 0.], [0., 0.]], -1),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[1., 2., math.nan]],
   [[1. / 4., 1. / 4.], [1. / 2., 0.], [0., 0.]], (0, 2)),
  ([[math.nan, math.nan], [math.nan, math.nan], [math.nan, math.nan]], [[math.nan, math.nan, math.nan]],
   [[0., 0.], [0., 0.], [0., 0.]], (0, 2)),
]


@pytest.mark.parametrize("scenario", SCENARIOS_MEAN_GRAD)
def test_reduce_nanmean_grad(scenario):
  x_list, y_list, g_list, dim = scenario
  x = torch.tensor([x_list, x_list], requires_grad=True)
  y = torch.tensor(y_list)
  g = torch.tensor([g_list, g_list])
  out = ReduceNanMean(dim=dim)(x)
  assert_near_nan(out, y)
  # Sum-of-outputs backward (upstream grad = 1 everywhere), matching TF's
  # tape.gradient(out, x) which implicitly does the same for a non-scalar out.
  # Plain .sum() (not a NaN-safe rewrite) - its backward is value-independent
  # (broadcast ones), so this exercises the real scenario (a downstream mean
  # over possibly-NaN per-slice results) without needing to work around
  # anything here; the safety has to come from ReduceNanMean's own custom
  # backward, matching TF's `tape.gradient(out, x)` on a non-scalar target.
  grads, = torch.autograd.grad(out.sum(), x)
  torch.testing.assert_close(grads, g, atol=1e-6, rtol=1e-6)


SCENARIOS_SUM = [
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], 8., None, None, math.nan),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], 12.,
   [[[1., 1.], [2., 2.], [1., 1.]], [[1., 1.], [2., 2.], [1., 1.]]], None, math.nan),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[2., 2., 0.], [2., 2., 0.]], None, -1, 0),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[2., 4., math.nan], [2., 4., math.nan]],
   [[1., 1.], [2., 2.], [1., 1.]], -1, math.nan),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[4., 4., math.nan]], None, (0, 2), math.nan),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[4., 8., math.nan]],
   [[[1., 1.], [2., 2.], [1., 1.]], [[1., 1.], [2., 2.], [1., 1.]]], (0, 2), math.nan),
  ([[math.nan, math.nan], [math.nan, math.nan], [math.nan, math.nan]], [[math.nan, math.nan, math.nan]],
   None, (0, 2), math.nan),
]


@pytest.mark.parametrize("scenario", SCENARIOS_SUM)
def test_reduce_nansum(scenario):
  x_list, y_list, weight_list, dim, default = scenario
  x = torch.tensor([x_list, x_list])
  y = torch.tensor(y_list)
  weight = torch.tensor(weight_list) if weight_list is not None else None
  assert_near_nan(reduce_nansum(x, weight=weight, dim=dim, default=default), y)


SCENARIOS_SUM_GRAD = [
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], 8., [[1., 1.], [1., 0.], [0., 0.]], None, None, math.nan),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], 12., [[1., 1.], [1., 0.], [0., 0.]],
   [[[1., 1.], [2., 2.], [1., 1.]], [[1., 1.], [2., 2.], [1., 1.]]], None, math.nan),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[2., 2., 0.], [2., 2., 0.]],
   [[1., 1.], [1., 0.], [0., 0.]], None, -1, 0),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[2., 4., math.nan], [2., 4., math.nan]],
   [[1., 1.], [1., 0.], [0., 0.]], [[1., 1.], [2., 2.], [1., 1.]], -1, math.nan),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[4., 4., math.nan]],
   [[1., 1.], [1., 0.], [0., 0.]], None, (0, 2), math.nan),
  ([[1., 1.], [2., math.nan], [math.nan, math.nan]], [[4., 8., math.nan]],
   [[1., 1.], [1., 0.], [0., 0.]], [[[1., 1.], [2., 2.], [1., 1.]], [[1., 1.], [2., 2.], [1., 1.]]], (0, 2), math.nan),
  ([[math.nan, math.nan], [math.nan, math.nan], [math.nan, math.nan]], [[math.nan, math.nan, math.nan]],
   [[0., 0.], [0., 0.], [0., 0.]], None, (0, 2), math.nan),
]


@pytest.mark.parametrize("scenario", SCENARIOS_SUM_GRAD)
def test_reduce_nansum_grad(scenario):
  x_list, y_list, g_list, weight_list, dim, default = scenario
  x = torch.tensor([x_list, x_list], requires_grad=True)
  y = torch.tensor(y_list)
  g = torch.tensor([g_list, g_list])
  weight = torch.tensor(weight_list) if weight_list is not None else None
  out = ReduceNanSum(weight=weight, dim=dim, default=default)(x)
  assert_near_nan(out, y)
  # Plain .sum() (not a NaN-safe rewrite) - its backward is value-independent
  # (broadcast ones), so this exercises the real scenario (a downstream mean
  # over possibly-NaN per-slice results) without needing to work around
  # anything here; the safety has to come from ReduceNanMean's own custom
  # backward, matching TF's `tape.gradient(out, x)` on a non-scalar target.
  grads, = torch.autograd.grad(out.sum(), x)
  torch.testing.assert_close(grads, g, atol=1e-6, rtol=1e-6)


SCENARIOS_LINEAR_COMBINATION = [
  ([.4, .7, 1.], [0.], [1., 2., .5], [.4, 1.4, .5], [1., 2., .5], [.6, .3, 0.], [.4, .7, 1.]),
  ([.4, .7, 1.], [0., 1., 1.], [1., 2., .5], [.4, 1.7, .5], [1., 1., -.5], [.6, .3, 0.], [.4, .7, 1.]),
  ([.4, .7, 1.], [0., math.nan, 1.], [1., 2., .5], [.4, math.nan, .5], [1., 0., -.5], [.6, .3, 0.], [.4, .7, 1.]),
]


@pytest.mark.parametrize("scenario", SCENARIOS_LINEAR_COMBINATION)
def test_nan_linear_combination(scenario):
  x_list, val_1_list, val_2_list, y_list, g_x_list, g_val_1_list, g_val_2_list = scenario
  x = torch.tensor(x_list, requires_grad=True)
  val_1 = torch.broadcast_to(torch.tensor(val_1_list), x.shape).clone().requires_grad_(True)
  val_2 = torch.broadcast_to(torch.tensor(val_2_list), x.shape).clone().requires_grad_(True)
  y = torch.tensor(y_list)
  g_x = torch.tensor(g_x_list)
  g_val_1 = torch.tensor(g_val_1_list)
  g_val_2 = torch.tensor(g_val_2_list)

  out = NanLinearCombination()(x, val_1, val_2)
  assert_near_nan(out, y)
  # Plain .sum() (see the analogous comment in test_reduce_nanmean_grad) -
  # TF's test also calls tape.gradient(out, ...) directly on a possibly-NaN
  # `out` with no workaround, relying on the fact that summation's own
  # backward is upstream-value-independent (broadcast ones) regardless of
  # whether the forward scalar it produces is itself NaN.
  grads = torch.autograd.grad(out.sum(), [x, val_1, val_2])
  torch.testing.assert_close(grads[0], g_x, atol=1e-6, rtol=1e-6)
  torch.testing.assert_close(grads[1], g_val_1, atol=1e-6, rtol=1e-6)
  torch.testing.assert_close(grads[2], g_val_2, atol=1e-6, rtol=1e-6)
