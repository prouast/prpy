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

import imghdr
import math
import numpy as np
import os
import logging
from typing import Union, Tuple

from prpy.ffmpeg.probe import probe_video
from prpy.ffmpeg.readwrite import read_video_from_path
from .image_ops import resample_bilinear_op, resample_box_op, reduce_roi_op

VIDEO_PARSE_ERROR = "Unable to parse input video. There may be an issue with the video file."

def _clip_roi(roi, h, w):
  """
  Clip an [x0, y0, x1, y1] ROI to the image rectangle [0, 0, w-1, h-1].

  Args:
    roi: Iterable with 4 coords.
    h, w: Image height and width.
  """
  roi = np.asarray(roi, dtype=float)
  # x coords -> [0, w], y coords -> [0, h]
  roi[[0, 2]] = np.clip(roi[[0, 2]], 0, w)
  roi[[1, 3]] = np.clip(roi[[1, 3]], 0, h)
  return roi.astype(int).tolist()

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
  """
  Crop, slice, and resize image(s) with all same settings.

  Args:
    inputs: The inputs of shape (h, w, 3) or (n_frames, h, w, 3)
    target_size: The target size; scalar or (h, w)
    roi: The region of interest in format (x0, y0, x1, y1). Use None to keep all.
    target_idxs: The frame indices to be used. Use None to keep all.
    preserve_aspect_ratio: Preserve the aspect ratio of inputs (or roi if provided) when scaling
    keepdims: If True, always keep n_frames dim. Otherwise, and when n_frames==1, drop n_frames dim.
    library: The library to use: `cv2`, `PIL`, `prpy` (return np.ndarray), or `tf` (returns tf.Tensor)
      - prpy is experimental and only supports bilinear scale_algorithm.
    scale_algorithm: The algorithm used for scaling.
      - Supports: bicubic, bilinear, area (not for PIL!), lanczos. Default: bicubic
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
  if roi is not None:
    h_img, w_img = inputs_shape[-3], inputs_shape[-2]
    roi = _clip_roi(roi=roi, h=h_img, w=w_img)
  # TODO: What if no target height but roi given?
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
        raise ValueError(f"Scaling algorithm {scale_algorithm} is not supported by {library}")
      out = tf.image.resize(
        images=inputs_sliced, size=out_size,
        preserve_aspect_ratio=preserve_aspect_ratio,
        method=library_algorithm, antialias=False)
    elif library == 'PIL':
      from PIL import Image
      # https://pillow.readthedocs.io/en/stable/releasenotes/2.7.0.html#image-resizing-filters
      mapping = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS}
      try:
        library_algorithm = mapping[scale_algorithm]
      except KeyError:
        raise ValueError(f"Scaling algorithm {scale_algorithm} is not supported by {library}")
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
        raise ValueError(f"Scaling algorithm {scale_algorithm} is not supported by {library}")
      # cv2 requires (width, height)
      out_size = (out_size[1], out_size[0])
      out = np.asarray([
        cv2.resize(src=f, dsize=out_size, interpolation=library_algorithm) for f in inputs_sliced])
    elif library == 'prpy':
      if scale_algorithm == 'bilinear':
        if inputs_sliced.shape[1] / out_size[0] > 2 and inputs_sliced.shape[2] / out_size[1] > 2:
          logging.debug("Switching from bilinear to box by default because we are downsampling significantly.")
          out = resample_box(im=inputs_sliced, size=out_size)
        else:
          out = resample_bilinear(im=inputs_sliced, size=out_size)
      else:
        raise ValueError(f"Scaling algorithm {scale_algorithm} is not supported by {library}")
    else:
      raise ValueError(f"Library {library} not supported")
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
  ) -> np.ndarray:
  """
  Compute bilinear resampling with batch dimension
  
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
  ) -> np.ndarray:
  """
  Compute box resampling with batch dimension

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
  """
  Reduce the spatial dimensions of a video by mean using ROI.

  Args:
    video: The video to be reduced. Shape (n, h, w, 3)
    roi: The roi in form (x0, y0, x1, y1). Shape (n, 4) or (4,)
  Returns:
    out: The reduced vals. Shape (n, 3)
  """
  if len(roi.shape) == 1:
    roi = np.tile(roi, (video.shape[0], 1))
  return reduce_roi_op(video, roi.astype(np.int64))

def probe_image_inputs(
    inputs: Union[np.ndarray, str],
    fps: float = None,
    min_video_frames: int = 1,
    allow_image: bool = True
  ) -> Tuple[tuple, float, bool]:
  """
  Check the image or video inputs and probe to extract metadata.

  Args:
    inputs: The inputs. Either
      - a filepath to video or image file
      - a `np.ndarray` of `np.uint8` representing a video or image with shape (n, h, w, c) or (h, w, c)
    fps: Sampling frequency of the input video. Required if type(video)==np.ndarray.
    allow_image: Whether to allow images. If False, raise ValueError on image inputs.
    min_frames: The minimum number of frames for a video
  Returns:
    Tuple of
     - shape: The shape of the input image or video as (n, h, w, c) or (h, w, c)
     - fps: Sampling frequency of the input video if video, else None.
     - issues: True if a possible issue with the inputs has been detected.
  """
  # Check if inputs is array or file name
  if isinstance(inputs, str):
    if os.path.isfile(inputs):
      # File
      if imghdr.what(inputs) is not None:
        # Image
        if not allow_image:
          raise ValueError(f"allow_image={allow_image}, but received a path to an image file.")
        from PIL import Image
        with Image.open(inputs) as img:
          width, height = img.size
          channels = len(img.getbands())
          return (height, width, channels), None, False
      else:
        # Video - check that fps is correct type
        if not (fps is None or isinstance(fps, (int, float))):
          raise ValueError(f"fps should be a number, but got {type(fps)}")
        try:
          fps_, n, w_, h_, _, _, r, i = probe_video(inputs)
          if fps is None: fps = fps_
          if abs(r) == 90: h = w_; w = h_
          else: h = h_; w = w_
          if not n >= min_video_frames:
            raise ValueError(f"video should have shape (n_frames [>= {min_video_frames}], h, w, 3), but found {(n, h, w, 3)}")
          return (n, h, w, 3), fps, i
        except Exception as e:
          raise ValueError(f"Problem probing video at {inputs}: {e}") 
    else:
      raise ValueError(f"No file found at {inputs}")
  elif isinstance(inputs, np.ndarray):
    # Array
    if len(inputs.shape) == 3:
      # Image
      if not allow_image:
        raise ValueError(f"allow_image={allow_image}, but received a ndarray with image data.")
      return inputs.shape, None, False
    elif len(inputs.shape) == 4:
      # Video - check that fps is correct type
      if not isinstance(fps, (int, float)):
        raise ValueError(f"fps should be specified as a number, but got {type(fps)}")
      if inputs.dtype != np.uint8:
        raise ValueError(f"video.dtype should be uint8, but got {inputs.dtype}")
      if inputs.shape[0] < min_video_frames or inputs.shape[3] != 3:
        raise ValueError(f"video should have shape (n_frames [>= {min_video_frames}], h, w, 3), but found {inputs.shape}")
      return inputs.shape, fps, False
    else:
      raise ValueError(f"Inputs should have rank 3 or 4, but had {len(inputs.shape)}")
  else:
    raise ValueError(f"Invalid video {inputs}, type {type(input)}")

def parse_image_inputs(
    inputs: Union[np.ndarray, str],
    fps: Union[float, int] = None,
    roi: Union[tuple, list] = None,
    target_size: Union[int, tuple, list] = None,
    target_fps: Union[float, int] = None,
    ds_factor: int = None,
    preserve_aspect_ratio: bool = False,
    library: str = 'prpy',
    scale_algorithm: str = 'bilinear',
    trim: Union[tuple, list] = None,
    allow_image: bool = True,
    videodims: bool = True 
  ) -> Tuple[np.ndarray, float, tuple, int, list]:
  """
  Parse image or video inputs into required shape.

  Args:
    inputs: The inputs. Either
      - a filepath to video or image file
      - a `np.ndarray` of `np.uint8` representing a video or image with shape (n, h, w, c) or (h, w, c)
    fps: Framerate for video inputs. Can be `None` if video file provided.
    roi: The region of interest as (x0, y0, x1, y1). Use None to keep all.
    target_size: Optional target size as int or tuple (h, w)
    target_fps: Optional target framerate for video.
    ds_factor: Optional alternative way to specify video downsampling - do not use both.
    preserve_aspect_ratio: Preserve the aspect ratio of inputs (or roi if provided) when scaling
    library: The library to use for scaling `np.ndarray` inputs.
      - `cv2`, `PIL`, `prpy` (return `np.ndarray`), or `tf` (returns `tf.Tensor`)
      - `prpy` is experimental and only supports bilinear scale_algorithm.
    scale_algorithm: The algorithm used for scaling.
      - Supports: bicubic, bilinear, area (not for PIL!), lanczos. Default: bicubic
    trim: Optional frame numbers for temporal trimming (start, end).
    allow_image: Allow image inputs or videos with one frame
    videodims: If True, always have four dims. Otherwise, and when n_frames==1, drop n_frames dim.
  Returns:
    Tuple of
     - parsed: Parsed inputs as `np.ndarray` with type uint8. Shape (n, h, w, c)
        if target_size provided, h = target_size[0] and w = target_size[1].
     - fps_in: Frame rate of original inputs if applicable, otherwise None
     - shape_in: Shape of original inputs in form (n, h, w, c) or (h, w, c)
     - ds_factor: Temporal downsampling factor applied if applicable, otherwise None
     - idxs: The frame indices returned from original inputs
  """
  assert (isinstance(inputs, np.ndarray) and (len(inputs.shape) == 3 or len(inputs.shape) == 4)) or (isinstance(inputs, str))
  assert fps is None or isinstance(fps, int) or isinstance(fps, float)
  assert roi is None or (isinstance(roi, (tuple, list)) and len(roi) == 4 and all(isinstance(i, (int, np.int64, np.int32)) for i in roi) and roi[2] > roi[0] and roi[3] > roi[1])
  assert target_size is None or isinstance(target_size, int) or (isinstance(target_size, (tuple, list)) and len(target_size) == 2 and all(isinstance(i, int) for i in target_size))
  assert target_fps is None or isinstance(target_fps, (float, int))
  assert (ds_factor is None or target_fps is None)
  assert ds_factor is None or isinstance(ds_factor, int)
  assert isinstance(preserve_aspect_ratio, bool)
  assert isinstance(library, str)
  assert isinstance(scale_algorithm, str)
  assert trim is None or (isinstance(trim, (tuple, list)) and len(trim) == 2 and all(isinstance(i, int) for i in trim))
  assert isinstance(allow_image, bool)
  assert isinstance(videodims, bool)
  def parse_np_image(inputs, roi, target_size, preserve_aspect_ratio, library, scale_algorithm):
    dims = inputs.shape
    if roi is not None or target_size is not None:
      if target_size is None and roi is not None: target_size = (int(roi[3]-roi[1]), int(roi[2]-roi[0]))
      elif target_size is None: target_size = (inputs.shape[0], inputs.shape[1])
      inputs = crop_slice_resize(
        inputs=inputs, target_size=target_size, roi=roi, target_idxs=None,
        preserve_aspect_ratio=preserve_aspect_ratio, library=library,
        scale_algorithm=scale_algorithm, keepdims=False)
    if videodims:
      inputs = inputs[np.newaxis]
      dims = (1,) + dims
    return inputs, None, dims, None, [0]
  # Check if input is array or file name
  if isinstance(inputs, str):
    if os.path.isfile(inputs):
      if imghdr.what(inputs) is not None:
        # Image file
        if not allow_image:
          raise ValueError(f"allow_image={allow_image}, but received a path to an image file.")
        try:
          from PIL import Image
          inputs = np.asarray(Image.open(open(inputs, 'rb')))
        except Exception as e:
          raise ValueError(f"Problem reading image from {inputs}: {e}")
        return parse_np_image(
          inputs=inputs, roi=roi, target_size=target_size, preserve_aspect_ratio=preserve_aspect_ratio,
          library=library, scale_algorithm=scale_algorithm)
      else:
        # Video file
        try:
          fps_, n, w_, h_, c, _, r, i = probe_video(inputs)
          if fps is None: fps = fps_
          if roi is not None: roi = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
          if isinstance(target_size, tuple): target_size = (target_size[1], target_size[0])
          if abs(r) == 90: h = w_; w = h_
          else: h = h_; w = w_
          if target_fps is None and ds_factor is not None: target_fps = fps / ds_factor
          try:
            inputs, ds_factor = read_video_from_path(
              path=inputs, target_fps=target_fps, crop=roi, scale=target_size, trim=trim,
              preserve_aspect_ratio=preserve_aspect_ratio, pix_fmt='rgb24', 
              scale_algorithm=scale_algorithm)
          except:
            raise ValueError(VIDEO_PARSE_ERROR)
          expected_n = math.ceil(((trim[1]-trim[0]) if trim is not None else n) / ds_factor)
          start_idx = max(0, trim[0]) if trim is not None else 0
          end_idx = min(n, trim[1]) if trim is not None else n
          idxs = list(range(start_idx, end_idx, ds_factor))
          if inputs.shape[0] != expected_n or len(idxs) != expected_n:
            raise ValueError(VIDEO_PARSE_ERROR)
          if not videodims:
            inputs = np.squeeze(inputs, axis=0)
          dims = (h, w, 3) if n == 1 and not videodims else (n, h, w, 3)
          return inputs, fps, dims, ds_factor, idxs
        except Exception as e:
          raise ValueError(f"Problem reading video from {inputs}: {e}")
    else:
      raise ValueError(f"No file found at {inputs}")
  elif isinstance(inputs, np.ndarray):
    shape_in = inputs.shape
    if roi is not None:
      h_img, w_img = shape_in[-3], shape_in[-2]
      roi = _clip_roi(roi=roi, h=h_img, w=w_img)
    if len(shape_in) == 3:
      # Image
      if not allow_image:
        raise ValueError(f"allow_image={allow_image}, but received a ndarray with image data.")
      return parse_np_image(
        inputs=inputs, roi=roi, target_size=target_size, preserve_aspect_ratio=preserve_aspect_ratio,
        library=library, scale_algorithm=scale_algorithm)
    else:
      # Video. Downsample / crop / scale if necessary
      if ds_factor is None: ds_factor = 1
      if target_fps is not None:
        if fps is None:
          raise ValueError("Must provide fps with `np.ndarray` video input and target_fps.")
        if target_fps > fps: logging.debug("target_fps should not be greater than fps. Ignoring.")
        else: ds_factor = max(round(fps / target_fps), 1)
      n = shape_in[0]
      expected_n = math.ceil(((trim[1]-trim[0]) if trim is not None else n) / ds_factor)
      target_idxs_start = 0 if trim is None else trim[0]
      target_idxs = None if ds_factor == 1 else list(range(target_idxs_start, n, ds_factor))
      if trim is not None:
        if target_idxs is None: target_idxs = range(target_idxs_start, n, 1)
        target_idxs = [idx for idx in target_idxs if trim[0] <= idx < trim[1]]
      if roi is not None or target_size is not None or target_idxs is not None:
        if target_size is None and roi is not None: target_size = (int(roi[3]-roi[1]), int(roi[2]-roi[0]))
        elif target_size is None: target_size = (shape_in[1], shape_in[2])
        inputs = crop_slice_resize(
          inputs=inputs, target_size=target_size, roi=roi, target_idxs=target_idxs,
          preserve_aspect_ratio=preserve_aspect_ratio, library=library, scale_algorithm=scale_algorithm)
      if target_idxs is None: target_idxs = list(range(shape_in[0]))
      if len(target_idxs) != expected_n or inputs.shape[0] != expected_n:
        logging.warning(f"Returning unexpected number of frames: {len(target_idxs)} instead of {expected_n}")
      return inputs, fps, shape_in, ds_factor, target_idxs
  else:
    raise ValueError(f"Invalid video {inputs}, type {type(inputs)}")
