###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, September 2021              #
###############################################################################

import numpy as np
import tensorflow as tf

def _reduction_dims(x, axis):
  """Resolve the reduction dims"""
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

def standardize_images(images, axis=None):
  """Standardize image data to have zero mean and unit variance.
  Args:
    images: The image data. 
    axis: The dimensions to standardize across. Exclude the axes that should be
      treated separately, e.g. the channel axis if desiring per-channel
      standardization. If None, standardize across all dimensions.
  Returns:
    images: The standardized image data as tf.float32.
  """
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

def normalize_images(images, axis=None):
  """Normalize image data to have zero mean.
  Args:
    images: The image data.
    axis: The dimensions to normalize across. Exclude the axes that should be
      treated separately, e.g. the channel axis if desiring per-channel
      normalization. If None, normalize across all dimensions.
  Returns:
    images: The normalized image data.
  """
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

def normalized_image_diff(images, axis=0):
  """Compute the normalized difference of adjacent images.
  Args:
    images: The image data as float32 in range [0, 1]
    axis: Scalar, the dimension across which to calculate difference
      normalization (e.g., the temporal/sequence dimension).
  Returns:
    images: The processed image data.
  """
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

def display_scale(image):
  """Scale any float32 image to the [0, 1] range for display"""
  min = tf.math.reduce_min(image)
  range = tf.math.reduce_max(image) - min
  return tf.clip_by_value((image-min) * 1.0/range, 0, 1)

def resize_with_random_method(image, target_shape=(640, 640)):
  """Resize an image to square with input_size"""
  # Draw a random number to determine resize method
  resize_method = tf.random.uniform([], 0, 5, dtype=tf.int32)
  def resize(method):
    def _resize():
      return tf.image.resize(
        image, target_shape, method=method, antialias=True, preserve_aspect_ratio=False)
    return _resize
  # Resize using a random method
  image = tf.case([(tf.equal(resize_method, 0), resize('bicubic')),
                   (tf.equal(resize_method, 1), resize('area')),
                   (tf.equal(resize_method, 2), resize('nearest')),
                   (tf.equal(resize_method, 3), resize('lanczos3'))],
                  default=resize('bilinear'))
  return image

def random_distortion(image):
  """Apply random distortion to image
  Args:
    image: The image [H, W, 3]
  Returns:
    image: The image [H, W, 3]
  """
  image = tf.image.random_brightness(image, 0.4)
  image = tf.image.random_contrast(image, 0.5, 1.5)
  image = tf.image.random_saturation(image, 0.5, 1.5)
  image = tf.image.random_hue(image, 0.1)
  return image
