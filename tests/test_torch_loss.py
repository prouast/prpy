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

"""Reuses the exact reference values from
tests/test_tensorflow.py::test_balanced_sample_weights."""

import torch

from prpy.torch.loss import balanced_sample_weights


def test_balanced_sample_weights_1d():
  out = balanced_sample_weights(
    labels=torch.tensor([1, 1, 2, 4, 1, 2]), unique=torch.arange(5))
  torch.testing.assert_close(
    out, torch.tensor([0.6666667, 0.6666667, 1., 2., 0.6666667, 1.]), atol=1e-6, rtol=1e-6)


def test_balanced_sample_weights_2d():
  out = balanced_sample_weights(
    labels=torch.tensor([[1], [1], [2], [4], [1], [2]]), unique=torch.arange(5))
  torch.testing.assert_close(
    out, torch.tensor([[0.6666667], [0.6666667], [1.], [2.], [0.6666667], [1.]]), atol=1e-6, rtol=1e-6)
