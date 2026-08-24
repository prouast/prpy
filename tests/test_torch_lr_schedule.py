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

"""Reuses the exact reference sequence from
tests/test_tensorflow.py::test_piecewise_constant_decay_with_warmup (already
verified against TF) as a value-only cross-framework check."""

import pytest

from prpy.torch.lr_schedule import piecewise_constant_decay_with_warmup

EXPECTED = (
  [0.01, 0.04, 0.07, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] +
  [0.01] * 10 +
  [0.001] * 10
)


def test_piecewise_constant_decay_with_warmup():
  lr_vals = [
    piecewise_constant_decay_with_warmup(
      i, boundaries=[9, 19], values=[0.1, 0.01, 0.001], warmup_init_lr=0.01, warmup_steps=3)
    for i in range(30)
  ]
  for actual, expected in zip(lr_vals, EXPECTED):
    assert actual == pytest.approx(expected, abs=1e-9)


def test_bad_boundaries_values_length_raises():
  with pytest.raises(ValueError):
    piecewise_constant_decay_with_warmup(
      0, boundaries=[9, 19], values=[0.1, 0.01], warmup_init_lr=0.01, warmup_steps=3)
