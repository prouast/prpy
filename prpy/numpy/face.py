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
from prpy.numpy.image import crop_slice_resize
from typing import Union

def _force_even_dims(roi: tuple) -> tuple:
  """Force even dimensions
  
  Args:
    roi: The region of interest with potentially uneven dimensions
  Returns:
    roi: The region of interest with forced even dimensions
  """
  roi_w = roi[2] - roi[0]
  roi_h = roi[3] - roi[1]
  if roi_w % 2 != 0:
    assert roi[2] > 2
    roi = (roi[0], roi[1], roi[2]-1, roi[3])
  if roi_h % 2 != 0:
    assert roi[3] > 2
    roi = (roi[0], roi[1], roi[2], roi[3]-1)
  return roi

def _get_roi_from_det(
    det: Union[tuple, np.ndarray],
    rel_change: tuple,
    clip_dims: Union[tuple, None] = None,
    force_even_dims: bool = False
  ) -> tuple:
  """Convert face detection to roi by relative add/reduce.

  Args:
    det: The face detection [0, H/W] in format (x0, y0, x1, y1).
    rel_change: The relative change to make in format (left, top, right, bottom).
    clip_dims: tuple (frame_w, frame_h) to clip the result to (optional).
    force_even_dims: Force to return even height and width roi.
  Returns:
    out: The roi [0, H/W] in format (x0, y0, x1, y1)
  """
  assert isinstance(det, (tuple, np.ndarray)) and len(det) == 4 and all(isinstance(i, (int, np.int64, np.int32)) for i in det)
  assert det[2] > det[0]
  assert det[3] > det[1]
  assert isinstance(rel_change, tuple) and len(rel_change) == 4 and all(isinstance(i, float) for i in rel_change)
  assert clip_dims is None or (isinstance(clip_dims, tuple) and len(clip_dims) == 2 and all(isinstance(i, int) for i in clip_dims))
  def _clip_dims(val, min_dim, max_dim):
    return min(max(val, min_dim), max_dim)
  det_w = det[2]-det[0]
  det_h = det[3]-det[1]
  rel_ch_l, rel_ch_t, rel_ch_r, rel_ch_b = rel_change
  abs_ch_l = int(rel_ch_l * det_w)
  abs_ch_t = int(rel_ch_t * det_h)
  abs_ch_r = int(rel_ch_r * det_w)
  abs_ch_b = int(rel_ch_b * det_h)
  if clip_dims is not None:
    out = (_clip_dims(det[0] - abs_ch_l, 0, clip_dims[0]),
           _clip_dims(det[1] - abs_ch_t, 0, clip_dims[1]),
           _clip_dims(det[2] + abs_ch_r, 0, clip_dims[0]),
           _clip_dims(det[3] + abs_ch_b, 0, clip_dims[1]))
  else:
    out = (det[0]-abs_ch_l, det[1]-abs_ch_t, det[2]+abs_ch_r, det[3]+abs_ch_b)
  return _force_even_dims(out) if force_even_dims else out

def get_face_roi_from_det(
    det: tuple,
    force_even_dims: bool = False
  ) -> tuple:
  """Convert face detection into face roi.
  Reduces width to 60% and height to 80%. 
  
  Args:
    det: The face detection [0, H/W] in form (x0, y0, x1, y1)
  Returns:
    out: The roi [0, H/W] in form (x0, y0, x1, y1)
  """
  return _get_roi_from_det(det=det,
                           rel_change=(-0.2, -0.1, -0.2, -0.1),
                           force_even_dims=force_even_dims)

def get_forehead_roi_from_det(
    det: tuple,
    force_even_dims: bool = False
  ) -> tuple:
  """Convert face detection into forehead roi.
  Reduces det to forehead as 35% to 65% of width, and 15% to 25% of height. 
  
  Args:
    det: The face detection [0, H/W] in form (x0, y0, x1, y1)
  Returns:
    out: The roi [0, H/W] in form (x0, y0, x1, y1)
  """
  return _get_roi_from_det(det=det,
                           rel_change=(-0.35, -0.15, -0.35, -0.75),
                           force_even_dims=force_even_dims)

def get_upper_body_roi_from_det(
    det: Union[tuple, np.ndarray],
    clip_dims: tuple,
    cropped: bool = False,
    v: int = 1,
    force_even_dims: bool = False
  ) -> tuple:
  """Convert face detection into upper body roi and clip to frame constraints.

  Args:
    det: The face detection [0, H/W] in form (x0, y0, x1, y1)
    clip_dims: constraints (frame_w, frame_h) to clip the result to
    cropped: Create cropped variant?
    v: Version of ROI definition (0, 1, 2, or 3)
  Returns:
    out: The roi [0, H/W] in form (x0, y0, x1, y1)
  """
  assert isinstance(cropped, bool)
  assert isinstance(v, int)
  if v == 0:
    # V0: (.25, .3, .25, .5) -> (.175, .27, .175, .45)
    if not cropped:
      return _get_roi_from_det(det=det,
                               rel_change=(.25, .3, .25, .5),
                               clip_dims=clip_dims,
                               force_even_dims=force_even_dims)
    else:
      return _get_roi_from_det(det=det,
                               rel_change=(.175, .27, .175, .45),
                               clip_dims=clip_dims,
                               force_even_dims=force_even_dims)
  elif v == 1:
    # V1: (.25, .2, .25, .4) -> (.175, .15, .175, .3)
    if not cropped:
      return _get_roi_from_det(det=det,
                               rel_change=(.25, .2, .25, .4),
                               clip_dims=clip_dims,
                               force_even_dims=force_even_dims)
    else:
      return _get_roi_from_det(det=det,
                               rel_change=(.175, .15, .175, .3), 
                               clip_dims=clip_dims,
                               force_even_dims=force_even_dims)
  elif v == 2:
    # V2: (.25, .1, .25, .5) -> (.175, .075, .175, .375)
    if not cropped:
      return _get_roi_from_det(det=det,
                               rel_change=(.25, .1, .25, .5),
                               clip_dims=clip_dims,
                               force_even_dims=force_even_dims)
    else:
      return _get_roi_from_det(det=det,
                               rel_change=(.175, .075, .175, .375),
                               clip_dims=clip_dims,
                               force_even_dims=force_even_dims)
  elif v == 3:
    # V3: (.2, .3, .2, .45) -> (.15, .25, .15, .35)
    if not cropped:
      return _get_roi_from_det(det=det,
                               rel_change=(.2, .3, .2, .45),
                               clip_dims=clip_dims,
                               force_even_dims=force_even_dims)
    else:
      return _get_roi_from_det(det=det,
                               rel_change=(.15, .25, .15, .35),
                               clip_dims=clip_dims,
                               force_even_dims=force_even_dims)
  else:
    raise ValueError("v {} is not defined".format(v))

def get_meta_roi_from_det(
    det: tuple,
    clip_dims: tuple,
    force_even_dims: bool = False
  ) -> tuple:
  """Convert face detection into meta roi and clip to frame constraints.

  Args:
    det: The face detection [0, H/W] in form (x0, y0, x1, y1)
    clip_dims: constraints (frame_w, frame_h) to clip the result to
  Returns:
    out: The roi [0, H/W] in form (x0, y0, x1, y1)
  """
  return _get_roi_from_det(det=det,
                           rel_change=(.2, .2, .2, .2),
                           clip_dims=clip_dims,
                           force_even_dims=force_even_dims)

def get_roi_from_det(
    det: tuple,
    roi_method: Union[str, None],
    clip_dims: Union[tuple, None] = None,
    force_even_dims: bool = False
  ) -> tuple:
  """Convert face detection into specified roi.

  Args:
    det: The face detection [0, H/W] in form (x0, y0, x1, y1)
    roi_method: Which roi method to use. Either 'forehead', 'face',
      'upper_body', 'upper_body_cropped', 'meta', None (directly use det)
    clip_dims: Constraints (frame_w, frame_h) to clip the result to (optional).
  Returns:
    out: The roi [0, H/W] in form (x0, y0, x1, y1)
  """
  assert roi_method is None or isinstance(roi_method, str)
  if roi_method == 'face':
    return get_face_roi_from_det(det,
                                 force_even_dims=force_even_dims)
  elif roi_method == 'forehead':
    return get_forehead_roi_from_det(det,
                                     force_even_dims=force_even_dims)
  elif roi_method == 'upper_body':
    assert clip_dims is not None
    return get_upper_body_roi_from_det(det,
                                       clip_dims=clip_dims,
                                       cropped=False,
                                       force_even_dims=force_even_dims)
  elif roi_method == 'upper_body_cropped':
    assert clip_dims is not None
    return get_upper_body_roi_from_det(det,
                                       clip_dims=clip_dims,
                                       cropped=True,
                                       force_even_dims=force_even_dims)
  elif roi_method == 'meta':
    assert clip_dims is not None
    return get_meta_roi_from_det(det,
                                 clip_dims=clip_dims,
                                 force_even_dims=force_even_dims)
  elif roi_method is None or roi_method == 'det':
    return _force_even_dims(det) if force_even_dims else det
  else:
    raise ValueError("roi method {} is not supported".format(roi_method))

def crop_resize_from_det(
    video: np.ndarray,
    det: tuple,
    size: tuple,
    roi_method: str,
    library: str,
    scale_algorithm: str,
    force_even_dims: bool = False
  ) -> np.ndarray:
  """Crop and resize a video according to a single face detection.
  Resize to specified size with specified method.

  Args:
    video: The video. Shape (n_frames, h, w, c)
    det: The face detection in form (x_0, y_0, x_1, y_1)
    size: The target size for resize - (h, w)
    roi_method: Which roi method to use. Either 'forehead', 'face',
      'upper_body', 'upper_body_cropped', 'meta', None (directly use det)
    library: The library used for resize (PIL, cv2, or tf - returns tf.Tensor)
    scale_algorithm: The algorithm used for scaling. Supports: bicubic,
      bilinear, area (not for PIL!), lanczos
  Returns:
    result: Cropped and resized video. Shape [n_frames, size[0], size[1], c]
  """
  assert isinstance(video, np.ndarray) and len(video.shape) == 4
  _, height, width, _ = video.shape
  roi = get_roi_from_det(det,
                         roi_method=roi_method,
                         clip_dims=(width, height),
                         force_even_dims=force_even_dims)
  return crop_slice_resize(
    inputs=video, target_size=size, roi=roi, library=library,
    preserve_aspect_ratio=False, scale_algorithm=scale_algorithm)
