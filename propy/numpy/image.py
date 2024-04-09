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
from typing import Union

def crop_slice_resize(
    inputs: np.ndarray,
    target_size: Union[int, tuple, list], 
    roi: Union[tuple, None] = None,
    target_idxs: Union[list, None] = None,
    preserve_aspect_ratio: bool = False,
    keepdims: bool = True,
    library: str = 'PIL',
    scale_algorithm: str = 'bicubic'
  ) -> np.ndarray:
  """Crop, slice, and resize image(s) with all same settings.

  Args:
    inputs: The inputs as uint8 shape (h, w, 3) or (n_frames, h, w, 3)
    target_size: The target size; scalar or (H, W) if preserve_aspect_ratio=False
    roi: The region of interest in format (x0, y0, x1, y1). Use None to keep all.
    target_idxs: The frame indices to be used. Use None to keep all.
    preserve_aspect_ratio: Preserve the aspect ratio?
    keepdims: If True, always keep n_frames dim. Otherwise, may drop n_frames dim.
    library: The library to use. `cv2` or `PIL` (return np ndarray), or `tf` (returns tf Tensor)
    scale_algorithm: The algorithm used for scaling.
      Supports: bicubic, bilinear, area (not for PIL!), lanczos. Default: bicubic
  Returns:
    result: The processed frame(s) as float32 shape (h, w, 3) or (n_frames, h, w, 3)
  """
  assert isinstance(inputs, np.ndarray) and (len(inputs.shape) == 3 or len(inputs.shape) == 4)
  assert isinstance(target_size, int) or (isinstance(target_size, (tuple, list)) and len(target_size) == 2 and all(isinstance(i, int) for i in target_size))
  assert roi is None or (isinstance(roi, tuple) and len(roi) == 4 and all(isinstance(i, int) for i in roi) and roi[2] > roi[0] and roi[3] > roi[1])
  assert target_idxs is None or (isinstance(target_idxs, list) and all(isinstance(i, int) for i in target_idxs))
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
  inputs = inputs[(target_idxs if target_idxs is not None else slice(None)), 
                  (slice(roi[1], roi[3]) if isinstance(roi, tuple) else slice(None)),
                  (slice(roi[0], roi[2]) if isinstance(roi, tuple) else slice(None))]
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
    if library == 'tf':
      import tensorflow as tf
      out = tf.convert_to_tensor(inputs, dtype=tf.float32)
    else:
      out = inputs.astype(np.float32) 
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
        images=inputs, size=(target_height, target_width),
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
        np.asarray(Image.fromarray(f).resize(out_size, resample=library_algorithm)) for f in inputs])
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
        cv2.resize(src=f, dsize=out_size, interpolation=library_algorithm) for f in inputs])
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
