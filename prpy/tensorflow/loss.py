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

def balanced_sample_weights(
    labels: tf.Tensor,
    unique: tf.Tensor
  ) -> tf.Tensor:
  """Calculate weights for a batch of dense categorical labels intended to be
  multiplied with the losses
  
  - Larger weights for examples of under-represented classes, and smaller weights
    for overrepresented classes, while keeping the total loss constant.
  - Important: Only works for dense label representation that equal range from 0 to n
  
  Args:
    labels: The dense categorical labels of shape (batch_size,) or (batch_size, 1)
    unique: The unique labels of shape (n_unique_labels,)
  Returns:
    weights: The weights with same shape as labels.
  """
  assert isinstance(labels, tf.Tensor)
  assert isinstance(unique, tf.Tensor)
  # Remove empty dim if necessary
  f_labels = tf.cast(tf.squeeze(labels), tf.int32)
  # Determine count of labels in batch
  # Cannot use bincount to be compatible with XLA
  # count = tf.math.bincount(f_labels, minlength=n_labels)
  def count(x):
    return tf.reduce_sum(tf.cast(tf.equal(x, f_labels), tf.int32))
  count = tf.map_fn(fn=count, elems=unique, fn_output_signature=tf.int32)
  # Batch size and number of unique labels
  batch_size = tf.size(f_labels)
  unique_count = tf.reduce_sum(tf.cast(tf.math.greater(count, 0), tf.int32))
  # Calculate the weight for each class
  class_weights = tf.math.divide_no_nan(tf.cast(batch_size, tf.float32), tf.cast(count, tf.float32))
  class_weights = class_weights / tf.cast(unique_count, tf.float32)
  # Gather weights according to the actual categories of the batch elements
  sample_weights = tf.gather(class_weights, f_labels)
  # Reshape to original shape
  sample_weights = tf.reshape(sample_weights, tf.shape(labels))
  return sample_weights

def smooth_l1_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    keepdims: bool = False
  ) -> tf.Tensor:
  """Smooth L1 loss

  Args:
    y_true: Labels. Shape arbitrary.
    y_pred: Predictions. Shape same as y_true.
    keepdims: Keep original shape? Otherwise return global mean.
  Returns:
    loss: The loss. Shape same as original if keepdims, otherwise ()
  """
  assert isinstance(y_true, tf.Tensor)
  assert isinstance(y_pred, tf.Tensor)
  assert isinstance(keepdims, bool)
  t = tf.abs(y_pred - y_true)
  loss = tf.where(t < 1, 0.5 * t ** 2, t - 0.5)
  return tf.cond(tf.equal(keepdims, tf.constant(True)),
    true_fn=lambda: loss,
    false_fn=lambda: tf.reduce_mean(loss))

def mae_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    keepdims: bool = False
  ) -> tf.Tensor:
  """Mean absolute error loss

  Args:
    y_true: Labels. Shape arbitrary.
    y_pred: Predictions. Shape same as y_true.
    keepdims: Keep batch dim? Otherwise return global mean.
  Returns:
    loss: The loss. Shape: If keepdims, original shape without final dim, otherwise ()
  """
  assert isinstance(y_true, tf.Tensor)
  assert isinstance(y_pred, tf.Tensor)
  assert isinstance(keepdims, bool)
  return tf.cond(tf.equal(keepdims, tf.constant(True)),
    true_fn=lambda: tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1),
    false_fn=lambda: tf.reduce_mean(tf.abs(y_pred - y_true)))
