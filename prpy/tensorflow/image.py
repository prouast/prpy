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

import numpy as np
import tensorflow as tf
from typing import Union

def _reduction_dims(
    x: tf.Tensor,
    axis: Union[int, tuple, None]
  ) -> Union[int, tuple, tf.Tensor]:
  """Resolve the reduction dims.

  - If axis is None, returns all dims of x for reduction
  
  Args:
    x: The tensor to be reduced
    axis: Dims to be reduced or None
  Returns:
    Either the axis, or a tensorflow equivalent
  """
  assert isinstance(x, tf.Tensor)
  assert axis is None or isinstance(axis, int) or isinstance(axis, tuple)
  if axis is not None:
    return axis
  else:
    x_rank = None
    if isinstance(x, tf.Tensor):
      x_rank = x.shape.rank
    # Fast path: avoid creating Rank and Range ops if ndims is known.
    if x_rank:
      return tf.constant(np.arange(x_rank, dtype=np.int32))
    else:
      # Otherwise, we rely on Range and Rank to do the right thing at run-time.
      return tf.range(0, tf.rank(x))

def standardize_images(
    images: Union[tf.Tensor, np.ndarray],
    axis: Union[int, tuple, None] = None
  ) -> tf.Tensor:
  """Standardize image data to have zero mean and unit variance.

  Args:
    images: The image data. 
    axis: The dimensions to standardize across. Exclude the axes that should be
      treated separately, e.g. the channel axis to treat each channel separately.
      If None, standardize across all dimensions.
  Returns:
    images: The standardized image data as tf.float32.
  """
  assert isinstance(images, (tf.Tensor, np.ndarray))
  assert axis is None or isinstance(axis, int) or isinstance(axis, tuple)
  # Convert to tf.Tensor if necessary
  if not tf.is_tensor(images):
    images = tf.convert_to_tensor(images)
  # We are working with tf.float32
  images = tf.cast(images, dtype=tf.float32)
  # Resolve axis arg
  axis = _reduction_dims(images, axis)
  # Compute the mean and std
  num_pixels = tf.math.reduce_prod(tf.gather(tf.shape(images), axis))
  mean = tf.math.reduce_mean(images, axis, keepdims=True)
  std = tf.math.reduce_std(images, axis, keepdims=True)
  # Apply a minimum normalization that protects us against uniform images
  min_std = tf.math.rsqrt(tf.cast(num_pixels, dtype=tf.float32))
  # Perform standardization
  images = tf.subtract(images, mean)
  images = tf.divide(images, tf.maximum(std, min_std))
  return images

def normalize_images(
    images: Union[tf.Tensor, np.ndarray],
    axis: Union[int, tuple, None] = None
  ) -> tf.Tensor:
  """Normalize image data to have zero mean.

  Args:
    images: The image data.
    axis: The dimensions to normalize across. Exclude the axes that should be
      treated separately, e.g. the channel axis to treat each channel separately.
      If None, normalize across all dimensions.
  Returns:
    images: The normalized image data.
  """
  assert isinstance(images, (tf.Tensor, np.ndarray))
  assert axis is None or isinstance(axis, int) or isinstance(axis, tuple)
  # Convert to tf.Tensor if necessary
  if not tf.is_tensor(images):
    images = tf.convert_to_tensor(images)
  # We are working with tf.float32
  images = tf.cast(images, dtype=tf.float32)
  # Resolve axis arg
  axis = _reduction_dims(images, axis)
  # Compute the mean
  mean = tf.math.reduce_mean(images, axis, keepdims=True)
  # Perform normalization
  images = tf.subtract(images, mean)
  return images

def normalized_image_diff(
    images: Union[tf.Tensor, np.ndarray],
    axis: int = 0
  ) -> tf.Tensor:
  """Compute the normalized difference of adjacent images.

  Args:
    images: The image data as float32 in range [0, 1]
    axis: Scalar, the dimension across which to calculate difference
      normalization (e.g., the temporal/sequence dimension).
  Returns:
    images: The processed image data.
  """
  assert isinstance(images, (tf.Tensor, np.ndarray))
  assert axis==0 or axis==1, "Only axis=0 or axis=1 supported"
  # Convert to tf.Tensor if necessary
  if not tf.is_tensor(images):
    images = tf.convert_to_tensor(images)
  # We are working with tf.float32
  images = tf.cast(images, dtype=tf.float32)
  diff = tf.cond(tf.equal(axis, 0),
    true_fn=lambda: images[1:] - images[:-1],
    false_fn=lambda: images[:,1:] - images[:,:-1])
  sum = tf.cond(tf.equal(axis, 0),
    true_fn=lambda: images[1:] + images[:-1],
    false_fn=lambda: images[:,1:] + images[:,:-1])
  sum = tf.clip_by_value(sum, clip_value_min=1e-7, clip_value_max=2)
  return diff / sum

def display_scale(
    image: tf.Tensor
  ) -> tf.Tensor:
  """Scale any float32 image to the [0, 1] range for display.
  
  Args:
    image: The image tensor.
  Returns:
    out: The scaled image for display.
  """
  assert isinstance(image, tf.Tensor)
  min = tf.math.reduce_min(image)
  range = tf.math.reduce_max(image) - min
  return tf.clip_by_value((image-min) * 1.0/range, 0, 1)

def resize_with_random_method(
    images: tf.Tensor,
    target_shape: Union[tuple, None] = (640, 640)
  ) -> tf.Tensor:
  """Resize image(s) with a random method.
  
  Args:
    images: Image data, either shape (b, h, w, c) or (h, w, c)
    target_shape: The resize shape in form (new_h, new_w)
  Returns:
    out: The resized image data with shape (b, new_h, new_w, c) or (new_h, new_w, c)
  """
  assert isinstance(images, tf.Tensor)
  assert target_shape is None or (isinstance(target_shape, tuple) and len(target_shape) == 2 and all(isinstance(i, int) for i in target_shape))
  # Draw a random number to determine resize method
  resize_method = tf.random.uniform([], 0, 5, dtype=tf.int32)
  def resize(method):
    def _resize():
      return tf.image.resize(
        images, target_shape, method=method, antialias=True, preserve_aspect_ratio=False)
    return _resize
  # Resize using a random method
  images = tf.case([(tf.equal(resize_method, 0), resize('bicubic')),
                   (tf.equal(resize_method, 1), resize('area')),
                   (tf.equal(resize_method, 2), resize('nearest')),
                   (tf.equal(resize_method, 3), resize('lanczos3'))],
                  default=resize('bilinear'))
  return images

def random_distortion(
    images: tf.Tensor
  ) -> tf.Tensor:
  """Apply random distortion to image(s)

  Args:
    images: The image data (b, h, w, c) or (h, w, c)
  Returns:
    images: The distorted image data (b, h, w, c) or (h, w, c)
  """
  assert isinstance(images, tf.Tensor)
  images = tf.image.random_brightness(images, 0.4)
  images = tf.image.random_contrast(images, 0.5, 1.5)
  images = tf.image.random_saturation(images, 0.5, 1.5)
  images = tf.image.random_hue(images, 0.1)
  return images
