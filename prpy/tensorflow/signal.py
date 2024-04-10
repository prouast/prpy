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
from typing import Union

def normalize(
    x: Union[tf.Tensor, list],
    axis: Union[int, tuple, None] = -1
  ) -> tf.Tensor:
  """Perform normalization

  Args:
    x: The input data
    axis: Axis over which to normalize
  Returns:
    x: The normalized data
  """
  assert isinstance(x, (tf.Tensor, list))
  assert axis is None or isinstance(axis, (int, tuple))
  # Convert to tf.Tensor if necessary
  if not tf.is_tensor(x):
    x = tf.convert_to_tensor(x)
  mean = tf.math.reduce_mean(x, axis=axis, keepdims=True)
  return x - mean

def scale(
    x: Union[tf.Tensor, list],
    axis: Union[int, tuple, None] = -1
  ) -> tf.Tensor:
  """Perform scaling

  Args:
    x: The input data
    axis: Axis over which to scale
  Returns:
    x: The scaled data
  """
  assert isinstance(x, (tf.Tensor, list))
  assert axis is None or isinstance(axis, (int, tuple))
  # Convert to tf.Tensor if necessary
  if not tf.is_tensor(x):
    x = tf.convert_to_tensor(x)
  std = tf.math.reduce_std(x, axis=axis, keepdims=True)
  return tf.math.divide_no_nan(x, std)

def standardize(
    x: Union[tf.Tensor, list],
    axis: Union[int, tuple, None] = -1
  ) -> tf.Tensor:
  """Perform standardization

  Args:
    x: The input data
    axis: Axis over which to standardize
  Returns:
    x: The standardized data
  """
  assert isinstance(x, (tf.Tensor, list))
  assert axis is None or isinstance(axis, (int, tuple))
  # Convert to tf.Tensor if necessary
  if not tf.is_tensor(x):
    x = tf.convert_to_tensor(x)
  mean = tf.math.reduce_mean(x, axis=axis, keepdims=True)
  std = tf.math.reduce_std(x, axis=axis, keepdims=True)
  return tf.math.divide_no_nan(x - mean, std)

def diff(
    x: Union[tf.Tensor, list],
    axis: int = 0):
  """Compute first signal difference.

  Args:
    x: The signal
    axis: Scalar, the dimension across which to calculate diff.
  Returns:
    y: The diff signal
  """
  assert isinstance(x, (tf.Tensor, list))
  # Convert to tf.Tensor if necessary
  if not tf.is_tensor(x):
    x = tf.convert_to_tensor(x)
  assert axis==0 or axis==1, "Only axis=0 or axis=1 supported"
  return tf.cond(tf.equal(axis, 0),
    true_fn=lambda: x[1:] - x[:-1],
    false_fn=lambda: x[:,1:] - x[:,:-1])
