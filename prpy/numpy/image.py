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
import logging
from typing import Union

from .image_ops import resample_bilinear_op, resample_box_op, reduce_roi_op

def crop_slice_resize(
    inputs: np.ndarray,
    target_size: Union[int, tuple, list], 
    roi: Union[tuple, list, None] = None,
    target_idxs: Union[tuple, list, np.ndarray, None] = None,
    preserve_aspect_ratio: bool = False,
    keepdims: bool = True,
    library: str = 'PIL',
    scale_algorithm: str = 'bicubic'
  ) -> np.ndarray:
  """Crop, slice, and resize image(s) with all same settings.

  Args:
    inputs: The inputs of shape (h, w, 3) or (n_frames, h, w, 3)
    target_size: The target size; scalar or (H, W) if preserve_aspect_ratio=False
    roi: The region of interest in format (x0, y0, x1, y1). Use None to keep all.
    target_idxs: The frame indices to be used. Use None to keep all.
    preserve_aspect_ratio: Preserve the aspect ratio? (must be False for library=prpy)
    keepdims: If True, always keep n_frames dim. Otherwise, may drop n_frames dim.
    library: The library to use: `cv2`, `PIL`, `prpy` (return np.ndarray), or `tf` (returns tf.Tensor)
      prpy is experimental and only supports bilinear scale_algorithm.
    scale_algorithm: The algorithm used for scaling.
      Supports: bicubic, bilinear, area (not for PIL!), lanczos. Default: bicubic
  Returns:
    result: The processed frame(s) of shape (h_new, w_new, 3) or (n_frames_new, h_new, w_new, 3)
  """
  assert isinstance(inputs, np.ndarray) and (len(inputs.shape) == 3 or len(inputs.shape) == 4)
  assert isinstance(target_size, int) or (isinstance(target_size, (tuple, list)) and len(target_size) == 2 and all(isinstance(i, int) for i in target_size))
  assert roi is None or (isinstance(roi, (tuple, list)) and len(roi) == 4 and all(isinstance(i, (int, np.int64, np.int32)) for i in roi) and roi[2] > roi[0] and roi[3] > roi[1])
  assert target_idxs is None or isinstance(target_idxs, np.ndarray) or (isinstance(target_idxs, (tuple, list)) and all(isinstance(i, int) for i in target_idxs))
  assert isinstance(preserve_aspect_ratio, bool)
  assert isinstance(keepdims, bool)
  assert isinstance(library, str)
  assert isinstance(scale_algorithm, str)
  unpack_target_size = lambda x: (x[0], x[1]) if isinstance(x, (list, tuple)) else (x, x)
  target_height, target_width = unpack_target_size(target_size)
  inputs_shape = inputs.shape
  # Add temporal dim if necessary
  if len(inputs_shape) == 3: inputs = inputs[np.newaxis,:,:,:]
  # Apply target_idxs and roi
  inputs_sliced = inputs[(target_idxs if target_idxs is not None else slice(None)), 
                         (slice(roi[1], roi[3]) if isinstance(roi, (tuple, list)) else slice(None)),
                         (slice(roi[0], roi[2]) if isinstance(roi, (tuple, list)) else slice(None))]
  in_shape = inputs_sliced.shape
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
    if library == 'tf':
      import tensorflow as tf
      out = tf.convert_to_tensor(inputs_sliced)
    else:
      out = inputs_sliced
  else:
    # Resize to out_size
    if library == 'tf':
      import tensorflow as tf
      # https://www.tensorflow.org/api_docs/python/tf/image/ResizeMethod
      mapping = {"bicubic": "bicubic", "bilinear": "bilinear", "lanczos": "lanczos3", "area": "area"}
      try:
        library_algorithm = mapping[scale_algorithm]
      except KeyError:
        raise ValueError("Scaling algorithm {} is not supported by {}".format(scale_algorithm, library))
      out = tf.image.resize(
        images=inputs_sliced, size=(target_height, target_width),
        preserve_aspect_ratio=preserve_aspect_ratio,
        method=library_algorithm, antialias=False)
    elif library == 'PIL':
      from PIL import Image
      # https://pillow.readthedocs.io/en/stable/releasenotes/2.7.0.html#image-resizing-filters
      mapping = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS}
      try:
        library_algorithm = mapping[scale_algorithm]
      except KeyError:
        raise ValueError("Scaling algorithm {} is not supported by {}".format(scale_algorithm, library))
      # PIL requires (width, height)
      out_size = (out_size[1], out_size[0])
      out = np.asarray([
        np.asarray(Image.fromarray(f).resize(out_size, resample=library_algorithm)) for f in inputs_sliced])
    elif library == 'cv2':
      import cv2
      # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
      mapping = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR, "lanczos": cv2.INTER_LANCZOS4, "area": cv2.INTER_AREA}
      try:
        library_algorithm = mapping[scale_algorithm]
      except KeyError:
        raise ValueError("Scaling algorithm {} is not supported by {}".format(scale_algorithm, library))
      # cv2 requires (width, height)
      out_size = (out_size[1], out_size[0])
      out = np.asarray([
        cv2.resize(src=f, dsize=out_size, interpolation=library_algorithm) for f in inputs_sliced])
    elif library == 'prpy':
      if preserve_aspect_ratio:
        raise ValueError("preserve_aspect_ratio=True is not supported for library=prpy")
      if scale_algorithm == 'bilinear':
        if inputs_sliced.shape[1] / target_height > 2 and inputs_sliced.shape[2] / target_width > 2:
          logging.debug("Switching from bilinear to box by default because we are downsampling significantly.")
          out = resample_box(im=inputs_sliced, size=(target_height, target_width))
        else:
          out = resample_bilinear(im=inputs_sliced, size=(target_height, target_width))
      else:
        raise ValueError("Scaling algorithm {} is not supported by {}".format(scale_algorithm, library))
    else:
      raise ValueError("Library {} not supported".format(library))
  if keepdims and len(out.shape) == 3 and len(inputs_shape) == 4:
    # Add temporal dim back - might have been lost when slicing
    newaxis = tf.newaxis if library == 'tf' else np.newaxis
    out = out[newaxis,:,:,:]
  elif not keepdims and len(out.shape) == 4:
    # Remove temporal dim if necessary
    squeeze = tf.squeeze if library == 'tf' else np.squeeze
    if out.shape[0] == 1:
      out = squeeze(out, axis=0)
  return out

def resample_bilinear(
    im: np.ndarray,
    size: Union[int, tuple]
  ):
  """Compute bilinear resampling with batch dimension
  
  Args:
    im: The image(s) to be resized. Shape (n, h, w, c) or (h, w, c)
    size: The new size either as scalar or (new_h, new_w)
  Returns:
    out: The resized image(s). Shape (n, new_h, new_w, c) or (new_h, new_w, c)
  """
  if isinstance(size, int): size = (size, size)
  return resample_bilinear_op(im, size)

def resample_box(
    im: np.ndarray,
    size: Union[int, tuple]
  ):
  """Compute box resampling with batch dimension

  - Note: This implementation only works for downsampling at least 2x in each dimension
    I.e., h/new_h >= 2 and w/new_w >= 2
  
  Args:
    im: The image(s) to be resized. Shape (n, h, w, c) or (h, w, c)
    size: The new size either as scalar or (new_h, new_w)
  Returns:
    out: The resized image(s). Shape (n, new_h, new_w, c) or (new_h, new_w, c)
  """
  if isinstance(size, int): size = (size, size)
  return resample_box_op(im, size)

def reduce_roi(
    video: np.ndarray,
    roi: np.ndarray
  ) -> np.ndarray:
  """Reduce the spatial dimensions of a video by mean using ROI.

  Args:
    video: The video to be reduced. Shape (n, h, w, 3)
    roi: The roi in form (x0, y0, x1, y1). Shape (n, 4) 
  Returns:
    out: The reduced vals. Shape (n, 3)
  """
  return reduce_roi_op(video, roi.astype(np.int64))
