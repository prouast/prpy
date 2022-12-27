###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import numpy as np

def crop_slice_resize(inputs, target_size, roi=None, target_idxs=None, preserve_aspect_ratio=False, keepdims=True, method='cv2'):
  """Crop, slice, and resize images with all same settings.
  Args:
    inputs: The inputs as uint8 shape [H, W, 3] or [n_frames, H, W, 3]
    target_size: The target size; scalar or shape (H, W) if preserve_aspect_ratio=False
    roi: The region of interest, shape [x0, y0, x1, y1]. Use None to keep all.
    target_idxs: The frame indices to be used (list). Use None to keep all.
    preserve_aspect_ratio: Preserve the aspect ratio?
    method: The method to use -> `cv2` or `PIL` (return np ndarray) or `tf` (returns tf Tensor) 
    keepdims: If True, always keep n_frames dim. Otherwise, may drop n_frames dim.
  Returns:
    result: The resized frames as float32 shape [H, W, 3] or [n_frames, H, W, 3]
  """
  unpack_target_size = lambda x: (x[0], x[1]) if isinstance(x, (list, tuple)) else (x, x)
  target_height, target_width = unpack_target_size(target_size)
  # Add temporal dim if necessary
  if len(inputs.shape) == 3: inputs = inputs[np.newaxis,:,:,:]
  # Apply roi
  if roi is not None:
    inputs = inputs[:,roi[1]:roi[3],roi[0]:roi[2]]
  # Apply target_idxs
  if target_idxs is not None:
    inputs = inputs[target_idxs]
  in_shape = inputs.shape
  # Compute out size
  def _out_size(in_shape, target_height, target_width, preserve_aspect_ratio):
    _, height, width, _ = in_shape
    if preserve_aspect_ratio:
      # Determine critical side
      inputs_r = float(width)/height
      target_r = float(target_width)/target_height
      if inputs_r < target_r:
        # Height is critical side
        out_size = (target_height, int(target_height*inputs_r))
      else:
        # Height is critical side
        out_size = (int(target_width/inputs_r), target_width)
    else:
      out_size = (target_height, target_width)
    return out_size
  out_size = _out_size(in_shape, target_height, target_width, preserve_aspect_ratio)
  # Distinguish between different cases
  if out_size == (in_shape[1], in_shape[2]):
    # No resizing necessary
    out = inputs.astype(np.float32)
  else:
    # Resize to out_size
    if method == 'tf':
      import tensorflow as tf
      out = tf.image.resize(
        images=inputs, size=(target_height, target_width),
        preserve_aspect_ratio=preserve_aspect_ratio, method='bilinear', antialias=False)
    elif method == 'PIL':
      from PIL import Image
      # PIL requires (width, height)
      out_size = (out_size[1], out_size[0])
      out = np.asarray([
        np.asarray(Image.fromarray(f).resize(out_size, resample=Image.BICUBIC)) for f in inputs])
    elif method == 'cv2':
      import cv2
      # cv2 requires (width, height)
      out_size = (out_size[1], out_size[0])
      out = np.asarray([
        cv2.resize(src=f, dsize=out_size, interpolation=cv2.INTER_AREA) for f in inputs])
    else:
      raise ValueError("Method {} not supported".format(method))
  if keepdims and len(out.shape) == 3:
    # Add temporal dim if necessary - might have been lost when slicing
    newaxis = tf.newaxis if method == 'tf' else np.newaxis
    out = out[newaxis,:,:,:]
  else:
    # Remove temporal dim if necessary
    squeeze = tf.squeeze if method == 'tf' else np.squeeze
    if out.shape[0] == 1:
      out = squeeze(out, axis=0)
  return out
