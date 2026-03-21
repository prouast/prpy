# Copyright (c) 2026 Philipp Rouast
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
import vitallens_core as vc

def get_face_roi_from_det(
    det: tuple,
    force_even_dims: bool = False
  ) -> tuple:
  """
  Convert face detection into face roi using vitallens-core.
  """
  return get_roi_from_det(det, roi_method='face', force_even_dims=force_even_dims)

def get_forehead_roi_from_det(
    det: tuple,
    force_even_dims: bool = False
  ) -> tuple:
  """
  Convert face detection into forehead roi using vitallens-core.
  """
  return get_roi_from_det(det, roi_method='forehead', force_even_dims=force_even_dims)

def get_upper_body_roi_from_det(
    det: Union[tuple, np.ndarray],
    clip_dims: tuple,
    cropped: bool = False,
    v: int = 1,
    force_even_dims: bool = False,
    detector: str = 'retinaface'
  ) -> tuple:
  """
  Convert face detection into upper body roi using vitallens-core.
  """
  assert isinstance(cropped, bool)
  roi_method = 'upper_body_cropped' if cropped else 'upper_body'
  return get_roi_from_det(det, roi_method=roi_method, clip_dims=clip_dims, force_even_dims=force_even_dims, detector=detector)

def get_meta_roi_from_det(
    det: tuple,
    clip_dims: tuple,
    force_even_dims: bool = False
  ) -> tuple:
  """
  Convert face detection into meta roi using vitallens-core.
  """
  return get_roi_from_det(det, roi_method='meta', clip_dims=clip_dims, force_even_dims=force_even_dims)

def get_roi_from_det(
    det: tuple,
    roi_method: Union[str, None],
    clip_dims: Union[tuple, None] = None,
    force_even_dims: bool = False,
    detector: str = 'retinaface'
  ) -> tuple:
  """
  Convert face detection into specified roi.

  Args:
    det: The face detection [0, H/W] in form (x0, y0, x1, y1)
    roi_method: Which roi method to use. Either 'forehead', 'face',
      'upper_body', 'upper_body_cropped', 'meta', None (directly use det)
    clip_dims: Constraints (frame_w, frame_h) to clip the result to (optional).
    force_even_dims: Force to return even height and width roi.
    detector: The detector used
  Returns:
    out: The roi [0, H/W] in form (x0, y0, x1, y1)
  """
  assert roi_method is None or isinstance(roi_method, str)
  if roi_method == 'meta':
      vc_method = (0.2, 0.2, 0.2, 0.2)
  elif roi_method is None or roi_method == 'det':
      vc_method = (0.0, 0.0, 0.0, 0.0)
  else:
      vc_method = roi_method
  vc_detector = 'apple_vision' if detector in ['apple_vision', 'applevision', 'vision'] else 'default'
  face_rect = vc.Rect(float(det[0]), float(det[1]), float(det[2]-det[0]), float(det[3]-det[1]))
  clip = (float(clip_dims[0]), float(clip_dims[1])) if clip_dims is not None else None
  res_rect = vc.calculate_roi(face_rect, vc_method, vc_detector, clip, force_even_dims)
  return (int(res_rect.x), int(res_rect.y), int(res_rect.x + res_rect.width), int(res_rect.y + res_rect.height))

def crop_resize_from_det(
    video: np.ndarray,
    det: tuple,
    size: tuple,
    roi_method: str,
    library: str,
    scale_algorithm: str,
    force_even_dims: bool = False,
    detector: str = 'retinaface'
  ) -> np.ndarray:
  """
  Crop and resize a video according to a single face detection.
  
  - Resize to specified size with specified method.

  Args:
    video: The video. Shape (n_frames, h, w, c)
    det: The face detection in form (x_0, y_0, x_1, y_1)
    size: The target size for resize - (h, w)
    roi_method: Which roi method to use. Either 'forehead', 'face',
      'upper_body', 'upper_body_cropped', 'meta', None (directly use det)
    library: The library used for resize (PIL, cv2, or tf - returns tf.Tensor)
    scale_algorithm: The algorithm used for scaling. Supports: bicubic,
      bilinear, area (not for PIL!), lanczos
    force_even_dims: Force to return even height and width roi.
    detector: The detector used
  Returns:
    result: Cropped and resized video. Shape [n_frames, size[0], size[1], c]
  """
  assert isinstance(video, np.ndarray) and len(video.shape) == 4
  _, height, width, _ = video.shape
  roi = get_roi_from_det(det,
                         roi_method=roi_method,
                         clip_dims=(width, height),
                         force_even_dims=force_even_dims,
                         detector=detector)
  return crop_slice_resize(
    inputs=video, target_size=size, roi=roi, library=library,
    preserve_aspect_ratio=False, scale_algorithm=scale_algorithm)
