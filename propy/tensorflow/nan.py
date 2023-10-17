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

def reduce_nansum(x, axis=None):
  """tf.reduce_sum, ignoring nan. Returns nan for all-nan slices.
  Args:
    x: The input tensor.
    axis: The dimension to reduce.
  Returns:
    The reduced tensor.
  """
  mask = tf.math.is_finite(x)
  sum = tf.reduce_sum(tf.where(mask, x, tf.zeros_like(x)), axis=axis)
  return sum
  