###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, October 2023                #
###############################################################################

import tensorflow as tf

def reduce_nanmean(x, axis=None):
  """tf.reduce_mean, ignoring nan. Returns nan for all-nan slices.
  Args:
    x: The input tensor.
    axis: The dimension to reduce.
  Returns:
    The reduced tensor.
  """
  mask = tf.math.is_finite(x)
  numerator = tf.reduce_sum(tf.where(mask, x, tf.zeros_like(x)), axis=axis)
  denominator = tf.reduce_sum(tf.cast(mask, dtype=x.dtype), axis=axis)
  return numerator / denominator

def reduce_nansum(x, weight=None, axis=None):
  """tf.reduce_sum, weighted by weight, ignoring nan. Returns nan for all-nan slices.
  Args:
    x: The input tensor.
    weight: The weight tensor, with the same shape as x.
    axis: The dimension to reduce.
  Returns:
    The reduced tensor.
  """
  mask = tf.math.is_finite(x)
  if weight is None:
    sum = tf.reduce_sum(tf.where(mask, x, tf.zeros_like(x)), axis=axis)
  else:
    weight = tf.where(mask, weight, tf.zeros_like(weight))
    sum = tf.reduce_sum(tf.where(mask, x * weight, tf.zeros_like(x)), axis=axis)
  # Check if all elements are NaN, and if so, return NaN
  result = tf.where(tf.reduce_all(tf.logical_not(mask), axis=axis),
                    tf.constant(float('nan'), dtype=tf.float32),
                    sum)
  return result
  