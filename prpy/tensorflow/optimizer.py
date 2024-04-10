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

from packaging import version
import tensorflow as tf

if version.parse(tf.__version__) <= version.parse("2.6.5"):
  from tensorflow.keras.optimizers import Adam # Legacy / V2 optimizer
  from tensorflow.keras.mixed_precision import LossScaleOptimizer
elif version.parse(tf.__version__) >= version.parse("2.9") and version.parse(tf.__version__) <= version.parse("2.11"):
  from keras.optimizers.optimizer_experimental.adam import Adam # Experimental / V3 optimizer
  from keras.optimizers.optimizer_experimental.adamw import AdamW # Experimental / V3 optimizer
  from keras.mixed_precision.loss_scale_optimizer import LossScaleOptimizerV3 as LossScaleOptimizer
elif version.parse(tf.__version__) > version.parse("2.11"):
  from tensorflow.keras.optimizers import Adam, AdamW # V3 optimizer
  from keras.mixed_precision.loss_scale_optimizer import LossScaleOptimizerV3 as LossScaleOptimizer
else:
  raise ImportError("This version of TensorFlow is not compatible.")

class EpochAdamMetaclass(type):
  """Metaclass that delegates EpochAdam instance creation."""
  def __call__(cls, **kwargs):
    if version.parse(tf.__version__) <= version.parse("2.6.5"):
      return EpochAdamV2(**kwargs)
    else:
      return EpochAdamExperimental(**kwargs)

class EpochAdamWMetaclass(type):
  """Metaclass that delegates EpochAdamW instance creation."""
  def __call__(cls, **kwargs):
    if version.parse(tf.__version__) > version.parse("2.9"):
      return EpochAdamWExperimental(**kwargs)
    else:
      raise ImportError("This version of TensorFlow is not compatible.")

class EpochAdam(metaclass=EpochAdamMetaclass):
  pass

class EpochAdamW(metaclass=EpochAdamWMetaclass):
  pass

class EpochAdamExperimental(Adam):
  """Experimental Adam optimizer that retrieves learning rate based on epochs"""
  def __init__(self, **kwargs):
    # Create epochs counter variable
    with tf.init_scope():
      # Lift the variable creation to init scope to avoid environment issue.
      self._epochs = tf.Variable(
        0, name="epochs", dtype=tf.int64, trainable=False)
    super().__init__(**kwargs)
    self._variables.append(self._epochs)
  @property
  def epochs(self):
    return self._epochs
  @epochs.setter
  def epochs(self, variable):
    if getattr(self, "_built", False):
      raise RuntimeError(
          "Cannot set `epochs` to a new Variable after "
          "the Optimizer weights have been created. Here it is "
          f"attempting to set `iterations` to {variable}."
          "Usually this means you are trying to set `iterations`"
          " after calling `apply_gradients()`. Please set "
          "`iterations` before calling `apply_gradients()`.")
    self._epochs = variable 
  def _build_learning_rate(self, learning_rate):
    with tf.init_scope():
      if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
        # Create a variable to hold the current learning rate.
        current_learning_rate = tf.convert_to_tensor(
            learning_rate(self.epochs))
        self._current_learning_rate = tf.Variable(
            current_learning_rate,
            name="current_learning_rate",
            dtype=current_learning_rate.dtype,
            trainable=False)
        return learning_rate
      return tf.Variable(
        learning_rate,
        name="learning_rate",
        dtype=tf.float32,
        trainable=False)
  def _compute_current_learning_rate(self):
    if isinstance(self._learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
      # Compute the current learning rate at the beginning of variable update.
      if hasattr(self, "_current_learning_rate"):
        self._current_learning_rate.assign(
            self._learning_rate(self.epochs))
      else:
        current_learning_rate = tf.convert_to_tensor(
          self._learning_rate(self.epochs))
        self._current_learning_rate = tf.Variable(
          current_learning_rate,
          name="current_learning_rate",
          dtype=current_learning_rate.dtype,
          trainable=False)
  def finish_epoch(self):
    """Increment epoch count and re-compute lr"""
    self._epochs.assign_add(1)
    self._compute_current_learning_rate()

class EpochAdamWExperimental(AdamW):
  """Experimental AdamW optimizer that retrieves learning rate based on epochs"""
  def __init__(self, **kwargs):
    # Create epochs counter variable
    with tf.init_scope():
      # Lift the variable creation to init scope to avoid environment issue.
      self._epochs = tf.Variable(
        0, name="epochs", dtype=tf.int64, trainable=False)
    super().__init__(**kwargs)
    self._variables.append(self._epochs)
  @property
  def epochs(self):
    return self._epochs
  @epochs.setter
  def epochs(self, variable):
    if getattr(self, "_built", False):
      raise RuntimeError(
          "Cannot set `epochs` to a new Variable after "
          "the Optimizer weights have been created. Here it is "
          f"attempting to set `iterations` to {variable}."
          "Usually this means you are trying to set `iterations`"
          " after calling `apply_gradients()`. Please set "
          "`iterations` before calling `apply_gradients()`.")
    self._epochs = variable 
  def _build_learning_rate(self, learning_rate):
    with tf.init_scope():
      if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
        # Create a variable to hold the current learning rate.
        current_learning_rate = tf.convert_to_tensor(
            learning_rate(self.epochs))
        self._current_learning_rate = tf.Variable(
            current_learning_rate,
            name="current_learning_rate",
            dtype=current_learning_rate.dtype,
            trainable=False)
        return learning_rate
      return tf.Variable(
        learning_rate,
        name="learning_rate",
        dtype=tf.float32,
        trainable=False)
  def _compute_current_learning_rate(self):
    if isinstance(self._learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
      # Compute the current learning rate at the beginning of variable update.
      if hasattr(self, "_current_learning_rate"):
        self._current_learning_rate.assign(
            self._learning_rate(self.epochs))
      else:
        current_learning_rate = tf.convert_to_tensor(
          self._learning_rate(self.epochs))
        self._current_learning_rate = tf.Variable(
          current_learning_rate,
          name="current_learning_rate",
          dtype=current_learning_rate.dtype,
          trainable=False)
  def finish_epoch(self):
    """Increment epoch count and re-compute lr"""
    self._epochs.assign_add(1)
    self._compute_current_learning_rate()

class EpochAdamV2(Adam):
  """V2 Adam optimizer that retrieves learning rate based on epochs"""
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

class EpochLossScaleOptimizerMetaclass(type):
  """Metaclass that delegates EpochLossScaleOptimizer instance creation."""
  def __call__(cls, inner_optimizer, **kwargs):
    if version.parse(tf.__version__) <= version.parse("2.6.5"):
      return EpochLossScaleOptimizerV2(inner_optimizer, **kwargs)
    else:
      return EpochLossScaleOptimizerV3(inner_optimizer, **kwargs)

class EpochLossScaleOptimizer(metaclass=EpochLossScaleOptimizerMetaclass):
  pass

class EpochLossScaleOptimizerV2(LossScaleOptimizer):
  """Subclass LossScaleOptimizer to use epochs. Works for tensorflow<=2.6.5"""
  @property
  def epochs(self):
    """Variable. The number of epochs."""
    return self._optimizer.epochs
  def finish_epoch(self):
    """Increment epoch count"""
    return self._optimizer.finish_epoch()

class EpochLossScaleOptimizerV3(LossScaleOptimizer):
  """Subclass LossScaleOptimizer to use epochs."""
  @property
  def epochs(self):
    """Variable. The number of epochs."""
    return self._optimizer.epochs
  def finish_epoch(self):
    """Increment epoch count"""
    return self._optimizer.finish_epoch()
