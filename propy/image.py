###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, September 2021              #
###############################################################################

import numpy as np
import tensorflow as tf

def _reduction_dims_tf(x, axis):
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
      return range(0, tf.rank(x))

def standardize_image_tf(images, axis=None):
  """Standardize image data to have zero mean and unit variance.
  Args:
    images: The image data.
    axis: The dimensions to standardize across. Exclude the axes that should be
      treated separately, e.g. the channel axis if desiring per-channel
      standardization. If None, standardize across all dimensions.
  Returns:
    images: The standardized image data.
  """
  # Resolve axis arg
  axis = _reduction_dims_tf(images, axis)
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

def normalize_image_tf(images, axis=None):
  """Normalize image data to have zero mean.
  Args:
    images: The image data.
    axis: The dimensions to normalize across. Exclude the axes that should be
      treated separately, e.g. the channel axis if desiring per-channel
      normalization. If None, normalize across all dimensions.
  Returns:
    images: The normalized image data.
  """
  # Resolve axis arg
  axis = _reduction_dims_tf(images, axis)
  # Compute the mean
  mean = tf.math.reduce_mean(images, axis, keepdims=True)
  # Perform normalization
  images = tf.subtract(images, mean)
  return images

def normalized_image_diff_tf(images, axis=0):
  """Compute the normalized difference of adjacent images.
  Args:
    images: The image data as float32 in range [0, 1]
    axis: Scalar, the dimension across which to calculate difference
      normalization.
  Returns:
    images: The processed image data.
  """
  def _diff(x, axis=0):
    nd = x.shape.rank
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    return x[slice1] - x[slice2]
  def _sum(x, axis=0):
    nd = x.shape.rank
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)
    return x[slice1] + x[slice2]
  nd = images.shape.rank
  images_diff = _diff(images, axis=axis)
  images_sum = _sum(images, axis=axis)
  images_sum = tf.clip_by_value(
    images_sum, clip_value_min=1e-7, clip_value_max=2)
  return images_diff / images_sum

def display_scale(image):
  """Scale any float32 image to the [0, 1] range for display"""
  min = tf.math.reduce_min(image)
  range = tf.math.reduce_max(image) - min
  return tf.clip_by_value((image-min) * 1.0/range, 0, 1)
