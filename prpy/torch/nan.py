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

from typing import Optional, Tuple, Union

import torch

Dims = Union[int, Tuple[int, ...], None]


def _as_dims(dim: Dims) -> Optional[Tuple[int, ...]]:
  if dim is None:
    return None
  return (dim,) if isinstance(dim, int) else tuple(dim)


def reduce_nanmean(x: torch.Tensor, dim: Dims = None) -> torch.Tensor:
  """torch.mean, ignoring non-finite vals.
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
  """torch.sum, weighted by weight, ignoring non-finite values.
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
  # Ascending order so earlier insertions don't shift later indices.
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
  - grad = 0 at non-finite input positions (including for all-non-finite slices)
  """
  def __init__(self, dim: Dims = None):
    self.dims = _as_dims(dim)
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return _ReduceNanMeanFn.apply(x, self.dims)


class _ReduceNanSumFn(torch.autograd.Function):
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
    # Deliberately does not multiply by weight - see ReduceNanSum's docstring.
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

  TODO(plethnet KNOWN_ISSUES.md #2): when `weight` is given, it scales the
  forward sum (`x * weight`) but is NOT applied in the gradient - matches a
  discrepancy in TF's own hand-written gradient, preserved here verbatim for
  parity with what plethnet_v4 was actually trained with. Don't fix without
  sign-off - it would change real training dynamics.
  """
  def __init__(self, weight: Optional[torch.Tensor] = None, dim: Dims = None, default: float = float('nan')):
    self.weight = weight
    self.dims = _as_dims(dim)
    self.default = default
  def __call__(self, x: torch.Tensor) -> torch.Tensor:
    return _ReduceNanSumFn.apply(x, self.weight, self.dims, self.default)


class _NanLinearCombinationFn(torch.autograd.Function):
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
  """Linear combination with one fixed element. Supports gradient.
  - Calculates (1 - x) * val_1 + x * val_2
  - Behavior when val_1 or val_2 are non-finite: out = nan, grad wrt x = 0
    (grad wrt val_1/val_2 are ordinary (1-x)/x)
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
