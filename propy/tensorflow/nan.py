###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, October 2023                #
###############################################################################

import tensorflow as tf

def reduce_nanmean(x, axis=None):
  """tf.reduce_mean, ignoring non-finite vals.
  - Returns nan for all-nan slices.
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

class ReduceNanMean:
  """tf.reduce_mean, ignoring non-finite values. Supports gradient.
    - Behavior when x is non-finite:
      out: Non-finite vals in a slice contribute 0
      out: All-non-finite slices are nan
      grad = 0
  """
  def __init__(self, axis=None):
    """Initialize.
    Args:
      axis: Axes to reduce by mean
    """
    self.axis = axis
  @tf.custom_gradient
  def __call__(self, x):
    """Compute the mean.
    Args:
      x: The values.
    Returns:
      out: The computed mean.
      grad: Function calculating the gradient
    """
    mask = tf.math.is_finite(x)
    num = tf.reduce_sum(tf.where(mask, x, tf.zeros_like(x)), axis=self.axis)
    den = tf.reduce_sum(tf.cast(mask, dtype=x.dtype), axis=self.axis)
    mean = num / den
    def grad(upstream):
      den = tf.reduce_sum(tf.cast(mask, dtype=x.dtype), axis=self.axis)
      # Tile upstream to match x
      if self.axis is not None:
        axis_list = list(self.axis) if isinstance(self.axis, tuple) else [self.axis]
        # Expand dims to match x 
        for axis in axis_list:
          upstream = tf.expand_dims(upstream, axis=axis)
          den = tf.expand_dims(den, axis=axis)
        # Tile
        tile_shape = [1 if s1 == s2 else s2 if s1 == 1 else s1 for s1, s2 in zip(x.shape, upstream.shape)]
        upstream = tf.tile(upstream, tile_shape)
        den = tf.tile(den, tile_shape)      
      # Compute gradient and set to 0 where input was not finite
      dout_dx = tf.where(mask, upstream / den, tf.zeros_like(x))
      return dout_dx
    return mean, grad

def reduce_nansum(x, weight=None, axis=None, default=float('nan')):
  """tf.reduce_sum, weighted by weight, ignoring non-finite values.
  - Returns default for all-nan slices.
  Args:
    x: The input tensor.
    weight: The weight tensor, with the same shape as x or broadcastable to it.
    axis: The dimension to reduce.
    default: The value to return for all-non-finite slices.
  Returns:
    The reduced tensor.
  """
  mask = tf.math.is_finite(x)
  if weight is None:
    sum = tf.reduce_sum(tf.where(mask, x, tf.zeros_like(x)), axis=axis)
  else:
    weight = tf.where(mask, weight, tf.zeros_like(weight))
    sum = tf.reduce_sum(tf.where(mask, x * tf.cast(weight, x.dtype), tf.zeros_like(x)), axis=axis)
  # If there are no finite elements in a slice, return the default.
  return tf.where(tf.reduce_all(tf.logical_not(mask), axis=axis),
                  tf.constant(default, dtype=x.dtype),
                  sum)

class ReduceNanSum:
  """tf.reduce_sum, weighted by weight, ignoring non-finite values. Supports gradient.
    - Behavior when x is non-finite:
      out: Non-finite vals in a slice contribute 0
      out: All-non-finite slices are set to default value
      grad = 0
  """
  def __init__(self, weight=None, axis=None, default=float('nan')):
    """Initialize.
    Args:
      weight: The weight tensor, with the same shape as x or broadcastable to it.
      axis: Axes to reduce by sum
      default: Value for all-non-finite slices
    """
    self.weight = weight
    self.axis = axis
    self.default = default
  @tf.custom_gradient
  def __call__(self, x):
    """Compute the sum.
    Args:
      x: The values.
    Returns:
      out: The computed sum.
      grad: Function calculating the gradient
    """
    mask = tf.math.is_finite(x)
    if self.weight is None:
      sum = tf.reduce_sum(tf.where(mask, x, tf.zeros_like(x)), axis=self.axis)
    else:
      weight = tf.where(mask, self.weight, tf.zeros_like(self.weight))
      sum = tf.reduce_sum(tf.where(mask, x * tf.cast(weight, x.dtype), tf.zeros_like(x)), axis=self.axis)
    # If there are no finite elements in a slice, return the default.
    out = tf.where(tf.reduce_all(tf.logical_not(mask), axis=self.axis),
                   tf.cast(self.default, x.dtype),
                   sum)
    def grad(upstream):
      # Tile upstream to match x
      if self.axis is not None:
        axis_list = list(self.axis) if isinstance(self.axis, tuple) else [self.axis]
        # Expand dims to match x 
        for axis in axis_list:
          upstream = tf.expand_dims(upstream, axis=axis)
        # Tile
        tile_shape = [1 if s1 == s2 else s2 if s1 == 1 else s1 for s1, s2 in zip(x.shape, upstream.shape)]
        upstream = tf.tile(upstream, tile_shape)
      # Set the gradient to 0 where input was not finite
      dout_dx = tf.where(mask, upstream, tf.zeros_like(x))
      return dout_dx
    return out, grad

class NanLinearCombination:
  """Linear combination with one fixed element. Supports gradient.
    - Calculates (1 - x) * val_1 + x * val_2
    - Behavior when val_1 or val_2 are non-finite:
      out = nan
      grad = 0
  """
  @tf.custom_gradient
  def __call__(self, x, val_1, val_2):
    """Compute the linear combination.
    Args:
      x: The combination weight. All elements must be in [0, 1]
      val_1: The second value used in the linear combination.
      val_2: The second value used in the linear combination.
    Returns:
      out: The computed sum.
      grad: Function calculating the gradient
    """
    # Compute the linear combination
    val_1 = tf.broadcast_to(val_1, x.shape)
    val_2 = tf.broadcast_to(val_2, x.shape)
    out = (1 - x) * val_1 + x * val_2
    def grad(upstream):
      mask = tf.math.logical_and(tf.math.is_finite(val_1), tf.math.is_finite(val_2))
      dout_dx = tf.where(mask, upstream * (val_2 - val_1), tf.zeros_like(x))
      dout_dval_1 = upstream * (1 - x)
      dout_dval_2 = upstream * x
      return dout_dx, dout_dval_1, dout_dval_2
    return out, grad
  