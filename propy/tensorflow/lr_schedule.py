###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import tensorflow as tf

class PiecewiseConstantDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Piecewise constant decay with warmup."""
  def __init__(self, boundaries, values, warmup_init_lr, warmup_steps, name=None):
    super(PiecewiseConstantDecayWithWarmup, self).__init__()
    if len(boundaries) != len(values) - 1:
        raise ValueError("The length of boundaries should be 1 less than the length of values")
    self.boundaries = boundaries
    self.values = values
    self.name = name
    self.warmup_steps = warmup_steps
    self.warmup_init_lr = warmup_init_lr
  def __call__(self, step):
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
