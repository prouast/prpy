###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import tensorflow as tf

class Adam(tf.keras.optimizers.Adam):
  """Adam optimizer that retrieves learning rate based on epochs"""
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self._epochs = None
  def _decayed_lr(self, var_dtype):
    """Get learning rate based on epochs."""
    lr_t = self._get_hyper("learning_rate", var_dtype)
    if isinstance(lr_t, tf.keras.optimizers.schedules.LearningRateSchedule):
      epochs = tf.cast(self.epochs, var_dtype)
      lr_t = tf.cast(lr_t(epochs), var_dtype)
    return lr_t
  @property
  def epochs(self):
    """Variable. The number of epochs."""
    if self._epochs is None:
      self._epochs = self.add_weight(
        "epochs", shape=[], dtype=tf.int64, trainable=False,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
      self._weights.append(self._epochs)
    return self._epochs
  def finish_epoch(self):
    """Increment epoch count"""
    return self._epochs.assign_add(1)

class LossScaleOptimizer(tf.keras.mixed_precision.LossScaleOptimizer):
  def _decayed_lr(self, var_dtype):
    """Get learning rate based on epochs."""
    return self._optimizer._decayed_lr(var_dtype)
  @property
  def epochs(self):
    """Variable. The number of epochs."""
    return self._optimizer.epochs
  def finish_epoch(self):
    """Increment epoch count"""
    return self._optimizer.finish_epoch()
