###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import tensorflow as tf

def balanced_sample_weights(labels, unique):
  """Calculate weights for a batch of dense categorical labels intended to be
     multiplied with the losses - larger weights for examples of under-
     represented classes, and smaller weights for overrepresented classes
     while keeping the total loss constant.
     Important: Only works for dense label representation that equal range from 0 to n
  Args:
    labels: The dense categorical labels of shape [batch_size] or [batch_size, 1]
    unique: The unique labels of shape [n_unique_labels,]
  Returns:
    weights: The weights with same shape as labels.
  """
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

def smooth_l1_loss(y_true, y_pred, keepdims=False):
  """Smooth L1 loss
  Args:
    y_true: Labels. Shape arbitrary.
    y_pred: Predictions. Shape same as y_true.
    keepdims: Keep original shape? Otherwise return global mean.
  Returns:
    loss: The loss. Shape same as original if keepdims, otherwise ()
  """
  t = tf.abs(y_pred - y_true)
  loss = tf.where(t < 1, 0.5 * t ** 2, t - 0.5)
  return tf.cond(tf.equal(keepdims, tf.constant(True)),
    true_fn=lambda: loss,
    false_fn=lambda: tf.reduce_mean(loss))

def mae_loss(y_true, y_pred, keepdims=False):
  """Mean absolute error loss
  Args:
    y_true: Labels. Shape arbitrary.
    y_pred: Predictions. Shape same as y_true.
    keepdims: Keep batch dim? Otherwise return global mean.
  Returns:
    loss: The loss. Shape: If keepdims, original shape without final dim, otherwise ()
  """
  return tf.cond(tf.equal(keepdims, tf.constant(True)),
    true_fn=lambda: tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1),
    false_fn=lambda: tf.reduce_mean(tf.abs(y_pred - y_true)))
