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

import tensorflow as tf

class EpochAdam(tf.keras.optimizers.Adam):
  """Adam optimizer that natively tracks epochs."""
  def __init__(self, learning_rate=0.001, **kwargs):
    super().__init__(learning_rate=learning_rate, **kwargs)
    with tf.init_scope():
      self._epochs = tf.Variable(
        initial_value=0, name="epochs", dtype=tf.int64, trainable=False
      )
  @property
  def epochs(self):
    return self._epochs
  @epochs.setter
  def epochs(self, variable):
    if self.built:
      raise RuntimeError(
        "Cannot set `epochs` to a new Variable after the Optimizer weights "
        "have been created. Please set `epochs` before calling `apply_gradients()`."
      )
    self._epochs = variable 
  def finish_epoch(self):
    """Increment epoch count."""
    self._epochs.assign_add(1)

class EpochAdamW(tf.keras.optimizers.AdamW):
  """AdamW optimizer that natively tracks epochs."""
  def __init__(self, learning_rate=0.001, weight_decay=0.004, **kwargs):
    super().__init__(learning_rate=learning_rate, weight_decay=weight_decay, **kwargs)
    with tf.init_scope():
      self._epochs = tf.Variable(
        initial_value=0, name="epochs", dtype=tf.int64, trainable=False
      )
  @property
  def epochs(self):
    return self._epochs
  @epochs.setter
  def epochs(self, variable):
    if self.built:
      raise RuntimeError(
        "Cannot set `epochs` to a new Variable after the Optimizer weights "
        "have been created. Please set `epochs` before calling `apply_gradients()`."
      )
    self._epochs = variable 
  def finish_epoch(self):
    """Increment epoch count."""
    self._epochs.assign_add(1)
