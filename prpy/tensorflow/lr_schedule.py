# Copyright (c) 2024 Philipp Rouast
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

import tensorflow as tf
from typing import Union

class PiecewiseConstantDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Piecewise constant decay with warmup."""
  def __init__(
      self,
      boundaries: list,
      values: list,
      warmup_init_lr: Union[float, int],
      warmup_steps: Union[int, tf.Variable],
      name: Union[str, None] = None
    ):
    super(PiecewiseConstantDecayWithWarmup, self).__init__()
    assert isinstance(boundaries, list)
    assert isinstance(values, list)
    assert isinstance(warmup_init_lr, (float, int))
    assert isinstance(warmup_steps, (int, tf.Variable))
    if len(boundaries) != len(values) - 1:
        raise ValueError("The length of boundaries should be 1 less than the length of values")
    self.boundaries = boundaries
    self.values = values
    self.name = name
    self.warmup_steps = warmup_steps
    self.warmup_init_lr = warmup_init_lr
  def __call__(
      self,
      step: int
    ) -> float:
    assert isinstance(step, (int, tf.Variable))
    with tf.name_scope(self.name or "PiecewiseConstantWarmUp"):
      step = tf.cast(tf.convert_to_tensor(step), tf.float32)
      pred_fn_pairs = []
      warmup_steps = self.warmup_steps
      boundaries = self.boundaries
      values = self.values
      warmup_init_lr = self.warmup_init_lr
      pred_fn_pairs.append((step <= warmup_steps, lambda: warmup_init_lr + step * (values[0] - warmup_init_lr) / warmup_steps))
      pred_fn_pairs.append((tf.logical_and(step <= boundaries[0], step > warmup_steps), lambda: tf.constant(values[0])))
      pred_fn_pairs.append((step > boundaries[-1], lambda: tf.constant(values[-1])))
      for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
        pred = (step > low) & (step <= high)
        pred_fn_pairs.append((pred, lambda v=v: tf.constant(v)))
      # The default isn't needed here because our conditions are mutually
      # exclusive and exhaustive, but tf.case requires it.
      return tf.case(pred_fn_pairs, lambda: tf.constant(values[0]), exclusive=True)
  def get_config(self):
    return {
      "boundaries": self.boundaries,
      "values": self.values,
      "warmup_steps": self.warmup_steps,
      "warmup_init_lr": self.warmup_init_lr,
      "name": self.name
    }
