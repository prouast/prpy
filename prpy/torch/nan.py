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

"""Torch port of prpy/tensorflow/nan.py.

Matches TF's split exactly: the plain lowercase functions (`reduce_nanmean`,
`reduce_nansum`) are value-only and NOT gradient-safe in either framework -
fine for metrics/eval, not for anything backpropagated through. The
`ReduceNanMean`/`ReduceNanSum`/`NanLinearCombination` classes are the
gradient-safe versions, each wrapping a `torch.autograd.Function` (torch's
equivalent of `tf.custom_gradient`) whose `backward` directly transcribes
TF's hand-written gradient formula.

Why a naive "mask before use, then let autograd differentiate normally"
rewrite is NOT sufficient here (worth recording, since it's the first thing
to try and it silently fails on one specific case): masking `x` before
squaring/summing correctly avoids the classic "double where" NaN-gradient
leak (`0 * nan == nan`) for elementwise ops. But `ReduceNanMean`'s forward is
`safe_x.sum() / mask.sum()`, and for an all-non-finite slice `mask.sum() ==
0`, so the forward value is (intentionally) `0/0 == nan`. Autograd's
backward rule for division still computes `d(out)/d(numerator) = 1/denom =
1/0 = inf` for that slice - a completely different mechanism from the
"double where" case (no masking trick prevents it, since it's the reduction
itself dividing by zero) - and `inf * 0` (from the numerator's own
mask-zeroed gradient) is `nan` again. This only bites the "entire slice is
NaN" case, which is exactly `test_reduce_nanmean_grad`'s 4th scenario below -
a naive rewrite passes every other case and fails only that one, so it's
worth stating explicitly rather than leaving as a trap for a future
"simplification". A custom backward sidesteps this because it computes
`grad_output / den` and immediately selects (via `where`, a plain forward
value pick, not a differentiated op) the zero branch wherever `mask` is
False - the `1/0 = inf` intermediate is discarded by value selection, never
by an autograd rule, so there is nothing for a second level of
differentiation to poison.
"""

from typing import Optional, Tuple, Union

import torch

Dims = Union[int, Tuple[int, ...], None]


def _as_dims(dim: Dims) -> Optional[Tuple[int, ...]]:
  if dim is None:
    return None
  return (dim,) if isinstance(dim, int) else tuple(dim)


def reduce_nanmean(x: torch.Tensor, dim: Dims = None) -> torch.Tensor:
  """torch.mean, ignoring non-finite vals. Value-only - see module docstring.
  - Returns `nan` for all-nan slices.
  Args:
    x: The input tensor.
    dim: The dimension(s) to reduce.
  Returns:
    The reduced tensor.
  """
  mask = torch.isfinite(x)
  safe_x = torch.where(mask, x, torch.zeros_like(x))
  dims = _as_dims(dim)
  numerator = safe_x.sum(dim=dims) if dims is not None else safe_x.sum()
  denominator = (mask.sum(dim=dims) if dims is not None else mask.sum()).to(x.dtype)
  return numerator / denominator


def reduce_nansum(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    dim: Dims = None,
    default: float = float('nan')
  ) -> torch.Tensor:
  """torch.sum, weighted by weight, ignoring non-finite values. Value-only -
  see module docstring.
  - Returns default for all-nan slices.
  Args:
    x: The input tensor.
    weight: The weight tensor, with the same shape as x or broadcastable to it.
    dim: The dimension(s) to reduce.
    default: The value to return for all-non-finite slices.
  Returns:
    The reduced tensor.
  """
  mask = torch.isfinite(x)
  safe_x = torch.where(mask, x, torch.zeros_like(x))
  if weight is not None:
    safe_weight = torch.where(mask, weight, torch.zeros_like(weight))
    safe_x = safe_x * safe_weight.to(x.dtype)
  dims = _as_dims(dim)
  total = safe_x.sum(dim=dims) if dims is not None else safe_x.sum()
  all_nan = (~mask).all(dim=dims) if dims is not None else (~mask).all()
  return torch.where(all_nan, torch.tensor(default, dtype=x.dtype, device=x.device), total)


def _unsqueeze_at(t: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
  """Expand-dims `t` at each of `dims` (as they'd appear in the un-reduced
  tensor), ascending order so earlier insertions don't shift later indices,
  matching the meaning of `dims` supplied by the caller.
  """
  for d in sorted(dims):
    t = t.unsqueeze(d)
  return t


class _ReduceNanMeanFn(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, dims):
    mask = torch.isfinite(x)
    safe_x = torch.where(mask, x, torch.zeros_like(x))
    numerator = safe_x.sum(dim=dims) if dims is not None else safe_x.sum()
    denominator = (mask.sum(dim=dims) if dims is not None else mask.sum()).to(x.dtype)
    ctx.save_for_backward(mask, denominator)
    ctx.dims = dims
    return numerator / denominator
  @staticmethod
  def backward(ctx, grad_output):
    mask, denominator = ctx.saved_tensors
    if ctx.dims is not None:
      grad_output = _unsqueeze_at(grad_output, ctx.dims)
      denominator = _unsqueeze_at(denominator, ctx.dims)
    dx = torch.where(mask, grad_output / denominator, torch.zeros_like(mask, dtype=grad_output.dtype))
    return dx, None


class ReduceNanMean:
  """torch.mean, ignoring non-finite values. Supports gradient.
  Behavior when x is non-finite:
  - out: Non-finite vals in a slice contribute 0
  - out: All-non-finite slices are nan
  - grad = 0 at non-finite input positions (including for all-non-finite
    slices - see module docstring for why this needs a custom backward)
  """
  def __init__(self, dim: Dims = None):
    self.dims = _as_dims(dim)
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return _ReduceNanMeanFn.apply(x, self.dims)


class _ReduceNanSumFn(torch.autograd.Function):
  """See ReduceNanSum's docstring: the backward here deliberately does NOT
  multiply by `weight`, matching a discrepancy in TF's own hand-written
  gradient that this port preserves verbatim rather than "fixing".
  """
  @staticmethod
  def forward(ctx, x, weight, dims, default):
    mask = torch.isfinite(x)
    safe_x = torch.where(mask, x, torch.zeros_like(x))
    if weight is not None:
      safe_weight = torch.where(mask, weight, torch.zeros_like(weight)).to(x.dtype)
      weighted = safe_x * safe_weight
    else:
      weighted = safe_x
    total = weighted.sum(dim=dims) if dims is not None else weighted.sum()
    all_nan = (~mask).all(dim=dims) if dims is not None else (~mask).all()
    out = torch.where(all_nan, torch.tensor(default, dtype=x.dtype, device=x.device), total)
    ctx.save_for_backward(mask)
    ctx.dims = dims
    return out
  @staticmethod
  def backward(ctx, grad_output):
    (mask,) = ctx.saved_tensors
    if ctx.dims is not None:
      grad_output = _unsqueeze_at(grad_output, ctx.dims)
    grad_output = torch.broadcast_to(grad_output, mask.shape)
    dx = torch.where(mask, grad_output, torch.zeros_like(mask, dtype=grad_output.dtype))
    return dx, None, None, None


class ReduceNanSum:
  """torch.sum, weighted by weight, ignoring non-finite values. Supports gradient.
  Behavior when x is non-finite:
  - out: Non-finite vals in a slice contribute 0
  - out: All-non-finite slices are set to default value
  - grad = 0 at non-finite input positions

  IMPORTANT, preserved from TF rather than a porting choice: when `weight` is
  given, it scales the *forward* sum (`x * weight`) but is NOT applied in the
  gradient - `d(out)/dx` is 1 (not `weight`) at every finite position. This
  mirrors what appears to be an unintentional discrepancy in TF's own
  hand-written custom gradient (its `grad()` closure never references
  `self.weight`). It matters in practice: plethnet's engine.py uses
  `ReduceNanSum(weight=loss_weights, ...)` to combine per-signal losses, so
  today, in production, each signal's configured `loss_weight` changes the
  *logged* combined loss value but does not actually reweight the gradient
  used to update the model - every signal gets equal gradient weight
  regardless of `loss_weight`. Kept exactly as-is for bit-for-bit parity with
  what plethnet_v4 was actually trained with; flagged here so it isn't
  mistaken for a torch-port bug, and isn't "fixed" without explicit sign-off
  (fixing it would change real training dynamics).
  """
  def __init__(self, weight: Optional[torch.Tensor] = None, dim: Dims = None, default: float = float('nan')):
    self.weight = weight
    self.dims = _as_dims(dim)
    self.default = default
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return _ReduceNanSumFn.apply(x, self.weight, self.dims, self.default)


class _NanLinearCombinationFn(torch.autograd.Function):
  """See module docstring for `ReduceNan{Mean,Sum}`'s custom-backward
  rationale; this one has an additional reason of its own: the forward value
  must propagate raw NaN exactly like TF's version whenever val_1/val_2 are
  non-finite (that's what lets a downstream ReduceNanMean/ReduceNanSum
  correctly exclude that batch element), so the forward pass cannot mask its
  inputs first - that would silently replace an intended NaN with a finite
  number. The gradient still needs asymmetric treatment: d(out)/dx must be
  clamped to 0 where val_1/val_2 are non-finite, while d(out)/d(val_1) and
  d(out)/d(val_2) are left as their ordinary, unmasked (1-x)/x values,
  matching TF exactly.
  """
  @staticmethod
  def forward(ctx, x, val_1, val_2):
    val_1 = torch.broadcast_to(val_1, x.shape)
    val_2 = torch.broadcast_to(val_2, x.shape)
    ctx.save_for_backward(x, val_1, val_2)
    return (1 - x) * val_1 + x * val_2
  @staticmethod
  def backward(ctx, grad_output):
    x, val_1, val_2 = ctx.saved_tensors
    mask = torch.isfinite(val_1) & torch.isfinite(val_2)
    grad_x = torch.where(mask, grad_output * (val_2 - val_1), torch.zeros_like(x))
    grad_val_1 = grad_output * (1 - x)
    grad_val_2 = grad_output * x
    return grad_x, grad_val_1, grad_val_2


class NanLinearCombination:
  """Linear combination with one fixed element.
  - Calculates (1 - x) * val_1 + x * val_2
  - Behavior when val_1 or val_2 are non-finite: out = nan, grad wrt x = 0
    (grad wrt val_1/val_2 are ordinary (1-x)/x, matching TF - see
    `_NanLinearCombinationFn` for why this needs a custom backward)
  """
  def __call__(self, x: torch.Tensor, val_1: torch.Tensor, val_2: torch.Tensor) -> torch.Tensor:
    """Compute the linear combination.
    Args:
      x: The combination weight. All elements must be in [0, 1]
      val_1: The first value used in the linear combination.
      val_2: The second value used in the linear combination.
    Returns:
      out: The computed linear combination, broadcast to x's shape.
    """
    return _NanLinearCombinationFn.apply(x, val_1, val_2)
