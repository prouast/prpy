###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import tensorflow as tf

def normalize(x, axis=-1):
  """Perform normalization
  Args:
    x: The input data
    axis: Axis over which to normalize
  Returns:
    x: The normalized data
  """
  # Convert to tf.Tensor if necessary
  if not tf.is_tensor(x):
    x = tf.convert_to_tensor(x)
  mean = tf.math.reduce_mean(x, axis=axis, keepdims=True)
  return x - mean

def standardize(x, axis=-1):
  """Perform standardization
  Args:
    x: The input data
    axis: Axis over which to standardize
  Returns:
    x: The standardized data
  """
  # Convert to tf.Tensor if necessary
  if not tf.is_tensor(x):
    x = tf.convert_to_tensor(x)
  mean = tf.math.reduce_mean(x, axis=axis, keepdims=True)
  std = tf.math.reduce_std(x, axis=axis, keepdims=True)
  return (x - mean) / std

def diff(x, axis=0):
  """Compute first signal difference.
  Args:
    x: The signal
    axis: Scalar, the dimension across which to calculate diff.
  Returns:
    y: The diff signal
  """
  # Convert to tf.Tensor if necessary
  if not tf.is_tensor(x):
    x = tf.convert_to_tensor(x)
  assert axis==0 or axis==1, "Only axis=0 or axis=1 supported"
  return tf.cond(tf.equal(axis, 0),
    true_fn=lambda: x[1:] - x[:-1],
    false_fn=lambda: x[:,1:] - x[:,:-1])
