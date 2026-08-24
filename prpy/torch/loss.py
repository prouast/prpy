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

"""Torch port of prpy/tensorflow/loss.py.

Only `balanced_sample_weights` is ported - it's the only function from this
module plethnet_tensorflow actually imports (used by engine.py's
`live_balanced` liveness-loss weighting). `smooth_l1_loss`/`mae_loss` are
unused there and have no bearing on this port.

TF's version routes around `tf.math.bincount` "to be compatible with XLA"
using a `tf.map_fn` loop instead - a constraint that doesn't apply to eager
torch code, so this uses `torch.bincount` directly.
"""

import torch


def balanced_sample_weights(labels: torch.Tensor, unique: torch.Tensor) -> torch.Tensor:
  """Calculate weights for a batch of dense categorical labels intended to be
  multiplied with the losses.
  - Larger weights for examples of under-represented classes, and smaller
    weights for overrepresented classes, while keeping the total loss constant.
  - Important: Only works for dense label representation that equal range from 0 to n.
  Args:
    labels: The dense categorical labels of shape (batch_size,) or (batch_size, 1)
    unique: The unique labels of shape (n_unique_labels,)
  Returns:
    weights: The weights with same shape as labels.
  """
  original_shape = labels.shape
  f_labels = labels.reshape(-1).long()
  minlength = int(unique.max().item()) + 1
  count = torch.bincount(f_labels, minlength=minlength)[unique.long()]
  batch_size = f_labels.numel()
  unique_count = (count > 0).sum()
  class_weights = torch.where(
    count > 0, batch_size / count.clamp(min=1).to(torch.float32), torch.zeros_like(count, dtype=torch.float32))
  class_weights = class_weights / unique_count.to(torch.float32)
  sample_weights = class_weights[f_labels]
  return sample_weights.reshape(original_shape)
