###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, January 2023                #
###############################################################################

import numpy as np
from propy.numpy.image import crop_slice_resize

def _get_roi_from_det(det, rel_change, clip_dims=None):
  """Convert face detection to roi by relative add/reduce.
  Args:
    det: The face detection [0, H/W]. Tuple (x0, y0, x1, y1)
    rel_change: The relative change to make. Tuple (left, top, right, bottom)
    clip_dims: None or tuple (frame_w, frame_h) to clip the result to.
  Returns:
    out: The roi [0, H/W]. Tuple (x0, y0, x1, y1)
  """
  assert det[2] > det[0]
  assert det[3] > det[1]
  def _clip_dims(val, min_dim, max_dim):
    return min(max(val, min_dim), max_dim)
  det_w = det[2]-det[0]
  det_h = det[3]-det[1]
  rel_ch_l, rel_ch_t, rel_ch_r, rel_ch_b = rel_change
  abs_ch_l = int(rel_ch_l * det_w)
  abs_ch_t = int(rel_ch_t * det_h)
  abs_ch_r = int(rel_ch_r * det_w)
  abs_ch_b = int(rel_ch_b * det_h)
  if clip_dims:
    return (_clip_dims(det[0] - abs_ch_l, 0, clip_dims[0]),
            _clip_dims(det[1] - abs_ch_t, 0, clip_dims[1]),
            _clip_dims(det[2] + abs_ch_r, 0, clip_dims[0]),
            _clip_dims(det[3] + abs_ch_b, 0, clip_dims[1]))
  else:
    return (det[0]-abs_ch_l, det[1]-abs_ch_t, det[2]+abs_ch_r, det[3]+abs_ch_b)

def get_face_roi_from_det(det):
  """Convert face detection into face roi.
     Reduces width to 60% and height to 80%. 
  Args:
    det: The face detection [0, H/W]. Tuple (x0, y0, x1, y1)
  Returns:
    out: The roi [0, H/W]. Tuple (x0, y0, x1, y1)
  """
  return _get_roi_from_det(det=det, rel_change=(-0.2, -0.1, -0.2, -0.1))

def get_forehead_roi_from_det(det):
  """Convert face detection into forehead roi.
     Reduces det to forehead as 35% to 65% of width, and 15% to 25% of height. 
  Args:
    det: The face detection [0, H/W]. Tuple (x0, y0, x1, y1)
  Returns:
    out: The roi [0, H/W]. Tuple (x0, y0, x1, y1)
  """
  return _get_roi_from_det(det=det, rel_change=(-0.35, -0.15, -0.35, -0.75))

def get_upper_body_roi_from_det(det, clip_dims, cropped=False):
  """Convert face detection into upper body roi and clip to frame constraints.
  Args:
    det: The face detection [0, H/W]. Tuple (x0, y0, x1, y1)
    clip_dims: None or tuple (frame_w, frame_h) to clip the result to.
  Returns:
    out: The roi [0, H/W]. Tuple (x0, y0, x1, y1)
  """
  # V0: (.25, .3, .25, .5) -> (.175, .27, .175, .45)
  # V1: (.25, .2, .25, .4) -> (.175, .15, .175, .3)
  # V2: (.25, .1, .25, .5) -> (.175, .075, .175, .375)
  # V3: (.2, .3, .2, .45) -> (.15, .25, .15, .35)
  if not cropped:
    return _get_roi_from_det(
      det=det, rel_change=(.2, .3, .2, .45), clip_dims=clip_dims)
  else:
    return _get_roi_from_det(
      det=det, rel_change=(.15, .25, .15, .35), clip_dims=clip_dims)

def get_meta_roi_from_det(det, clip_dims):
  """Convert face detection into meta roi and clip to frame constraints.
  Args:
    det: The face detection [0, H/W]. Tuple (x0, y0, x1, y1)
    clip_dims: None or tuple (frame_w, frame_h) to clip the result to.
  Returns:
    out: The roi [0, H/W]. Tuple (x0, y0, x1, y1)
  """
  return _get_roi_from_det(
    det=det, rel_change=(.2, .2, .2, .2), clip_dims=clip_dims)

def get_roi_from_det(det, roi_method, clip_dims=None):
  """Convert face detection into specified roi.
  Args:
    det: The face detection [0, H/W]. Tuple (x0, y0, x1, y1)
    roi_method: Which roi method to use. Use 'forehead', 'face', 'upper_body', 'upper_body_cropped', None (directly use det)
    clip_dims: None or tuple (frame_w, frame_h) to clip the result to.
  Returns:
    out: The roi [0, H/W]. Tuple (x0, y0, x1, y1)
  """
  if roi_method == 'face':
    return get_face_roi_from_det(det)
  elif roi_method == 'forehead':
    return get_forehead_roi_from_det(det)
  elif roi_method == 'upper_body':
    assert clip_dims is not None
    return get_upper_body_roi_from_det(det, clip_dims=clip_dims, cropped=False)
  elif roi_method == 'upper_body_cropped':
    assert clip_dims is not None
    return get_upper_body_roi_from_det(det, clip_dims=clip_dims, cropped=True)
  elif roi_method == 'meta':
    assert clip_dims is not None
    return get_meta_roi_from_det(det, clip_dims=clip_dims)
  elif roi_method is None or roi_method == 'det':
    return det
  else:
    raise ValueError("roi method {} is not supported".format(roi_method))

def crop_resize_from_det(video, det, size, roi_method, method):
  """Crop and resize a video. Crop for face roi based on a single detection.
  Resize to specified size with specified method.
  Args:
    video: The video. Shape [n_frames, h, w, c]
    det: The face detection. Shape [4]
    size: The target size for resize - tuple (h, w)
    roi_method: Which roi method to use. Use 'forehead', 'face', 'upper_body', 'meta', None (directly use det)
    method: The resize method (tf, PIL, or cv2)
  Returns:
    result: Cropped and resized video. Shape [n_frames, size[0], size[1], c]
  """
  _, height, width, _ = video.shape
  roi = get_roi_from_det(det, roi_method=roi_method, clip_dims=(width, height))
  roi = np.asarray(roi, dtype=np.int64)
  return crop_slice_resize(
    inputs=video, target_size=size, roi=roi, method=method, preserve_aspect_ratio=False)
