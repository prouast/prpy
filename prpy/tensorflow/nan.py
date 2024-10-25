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
from typing import Union, Tuple, Callable

def reduce_nanmean(
    x: tf.Tensor,
    axis: Union[int, tuple, None] = None
  ) -> tf.Tensor:
  """tf.reduce_mean, ignoring non-finite vals.

  - Returns `nan` for all-nan slices.
  
  Args:
    x: The input tensor.
    axis: The dimension to reduce.
  Returns:
    The reduced tensor.
  """
  assert isinstance(x, tf.Tensor)
  assert axis is None or isinstance(axis, (int, tuple))
  mask = tf.math.is_finite(x)
  numerator = tf.reduce_sum(tf.where(mask, x, tf.zeros_like(x)), axis=axis)
  denominator = tf.reduce_sum(tf.cast(mask, dtype=x.dtype), axis=axis)
  return numerator / denominator

class ReduceNanMean:
  """tf.reduce_mean, ignoring non-finite values. Supports gradient.

  Behavior when x is non-finite:
  - out: Non-finite vals in a slice contribute 0
  - out: All-non-finite slices are nan
  - grad = 0
  """
  def __init__(self, axis: Union[int, tuple, None] = None):
    """Initialize.

    Args:
      axis: Axes to reduce by mean
    """
    assert axis is None or isinstance(axis, int) or (isinstance(axis, tuple) and all(isinstance(i, int) for i in axis))
    self.axis = axis
  @tf.custom_gradient
  def __call__(self, x: tf.Tensor) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
    """Compute the mean.

    Args:
      x: The values.
    Returns:
      Tuple of
       - out: The computed mean.
       - grad: Function calculating the gradient
    """
    assert isinstance(x, tf.Tensor)
    mask = tf.math.is_finite(x)
    num = tf.reduce_sum(tf.where(mask, x, tf.zeros_like(x)), axis=self.axis)
    den = tf.reduce_sum(tf.cast(mask, dtype=x.dtype), axis=self.axis)
    mean = num / den
    def grad(upstream: tf.Tensor) -> tf.Tensor:
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

def reduce_nansum(
    x: tf.Tensor,
    weight: Union[tf.Tensor, list, None] = None,
    axis: Union[int, tuple, None] = None,
    default: Union[float, int] = float('nan')
  ) -> tf.Tensor:
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
  assert isinstance(x, tf.Tensor)
  assert weight is None or isinstance(weight, (tf.Tensor, list))
  assert axis is None or isinstance(axis, (int, tuple))
  assert isinstance(default, (float, int))
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

  Behavior when x is non-finite:
  - out: Non-finite vals in a slice contribute 0
  - out: All-non-finite slices are set to default value
  - grad = 0
  """
  def __init__(
      self,
      weight: Union[tf.Tensor, list, None] = None,
      axis: Union[int, tuple, None] = None,
      default: Union[int, float] = float('nan')
    ):
    """Initialize.

    Args:
      weight: The weight tensor, with the same shape as x or broadcastable to it.
      axis: Axes to reduce by sum
      default: Value for all-non-finite slices
    """
    assert weight is None or isinstance(weight, (tf.Tensor, list))
    assert axis is None or isinstance(axis, (int, tuple))
    assert isinstance(default, (int, float))
    self.weight = weight
    self.axis = axis
    self.default = default
  @tf.custom_gradient
  def __call__(self, x: tf.Tensor) -> Tuple[tf.Tensor, Callable[[tf.Tensor], tf.Tensor]]:
    """Compute the sum.

    Args:
      x: The values.
    Returns:
      Tuple of
       - out: The computed sum.
       - grad: Function calculating the gradient
    """
    assert isinstance(x, tf.Tensor)
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
    def grad(upstream: tf.Tensor) -> tf.Tensor:
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
    - out = nan
    - grad = 0
  """
  @tf.custom_gradient
  def __call__(
    self,
    x: tf.Tensor,
    val_1: tf.Tensor,
    val_2: tf.Tensor
  ) -> Tuple[tf.Tensor, Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]]:
    """Compute the linear combination.

    Args:
      x: The combination weight. All elements must be in [0, 1]
      val_1: The second value used in the linear combination.
      val_2: The second value used in the linear combination.
    Returns:
      Tuple of
       - out: The computed sum.
       - grad: Function calculating the gradient
    """
    # Compute the linear combination
    val_1 = tf.broadcast_to(val_1, x.shape)
    val_2 = tf.broadcast_to(val_2, x.shape)
    out = (1 - x) * val_1 + x * val_2
    def grad(upstream: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
      mask = tf.math.logical_and(tf.math.is_finite(val_1), tf.math.is_finite(val_2))
      dout_dx = tf.where(mask, upstream * (val_2 - val_1), tf.zeros_like(x))
      dout_dval_1 = upstream * (1 - x)
      dout_dval_2 = upstream * x
      return dout_dx, dout_dval_1, dout_dval_2
    return out, grad
  