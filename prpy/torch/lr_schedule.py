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

"""Torch port of prpy/tensorflow/lr_schedule.py.

TF's `PiecewiseConstantDecayWithWarmup` is a `LearningRateSchedule` object
because Keras optimizers expect one (evaluated lazily, inside the graph, at
`optimizer.iterations`) - which is precisely the trap plethnet_tensorflow's
own engine.py works around (see its long comment on epoch-vs-iteration
scheduling): it constructs the object but then calls it manually as a plain
function of *epoch*, once per epoch, via `optimizer.learning_rate.assign(
lr_schedule(epoch))`, never letting Keras call it automatically.

Since torch's optimizers don't have a competing automatic-call mechanism -
`torch.optim.Optimizer` has no built-in notion of "schedule" at all, you
always drive `param_group['lr']` yourself - there's no analogous trap to
work around, and no need for a class wrapping graph ops (`tf.case`,
`tf.name_scope`) either. This is a plain Python function: call it once per
epoch and assign the result directly, e.g.
`for g in optimizer.param_groups: g['lr'] = piecewise_constant_decay_with_warmup(epoch, ...)`.
"""

from typing import List, Union


def piecewise_constant_decay_with_warmup(
    step: Union[int, float],
    boundaries: List[Union[int, float]],
    values: List[float],
    warmup_init_lr: float,
    warmup_steps: Union[int, float]
  ) -> float:
  """Piecewise constant decay with linear warmup.
  Args:
    step: The current step (or epoch - this function doesn't care about the
      unit, it just compares `step` against `boundaries`/`warmup_steps` in
      whatever unit they're expressed in. plethnet's engine calls this once
      per epoch with `step=epoch`, matching how it constructs `boundaries`).
    boundaries: Step boundaries. len(boundaries) == len(values) - 1.
    values: The constant LR value for each segment between boundaries.
    warmup_init_lr: The initial LR at step 0, ramping linearly to values[0]
      by `warmup_steps`.
    warmup_steps: Number of steps for the warmup ramp.
  Returns:
    The learning rate at `step`.
  """
  if len(boundaries) != len(values) - 1:
    raise ValueError("The length of boundaries should be 1 less than the length of values")
  if step <= warmup_steps:
    return warmup_init_lr + step * (values[0] - warmup_init_lr) / warmup_steps
  if step <= boundaries[0]:
    return values[0]
  if step > boundaries[-1]:
    return values[-1]
  for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
    if low < step <= high:
      return v
  # Unreachable given the checks above (boundaries/values are consistent by
  # construction), but avoids an implicit `None` return if that invariant is
  # ever violated by a caller.
  raise ValueError(f"step={step} did not match any segment of boundaries={boundaries}")
