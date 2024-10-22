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
import pytest

import sys
sys.path.append('../prpy')

from prpy.numpy.face import get_face_roi_from_det, get_forehead_roi_from_det, get_meta_roi_from_det
from prpy.numpy.face import get_upper_body_roi_from_det, get_roi_from_det, crop_resize_from_det

def test_get_face_roi_from_det():
  det = (100, 100, 180, 220)
  roi = get_face_roi_from_det(det)
  assert roi == (116, 112, 164, 208)
  det = (100, 100, 181, 220)
  roi = get_face_roi_from_det(det, force_even_dims=True)
  assert roi == (116, 112, 164, 208)

def test_get_forehead_roi_from_det():
  det = (100, 100, 180, 220)
  roi = get_forehead_roi_from_det(det)
  assert roi == (128, 118, 152, 130)
  det = (100, 100, 181, 220)
  roi = get_forehead_roi_from_det(det, force_even_dims=True)
  assert roi == (128, 118, 152, 130)

@pytest.mark.parametrize("cropped", [True, False])
@pytest.mark.parametrize("v", [0, 1, 2, 3])
def test_get_upper_body_roi_from_det(cropped, v):
  det = (100, 100, 180, 220)
  roi = get_upper_body_roi_from_det(det, clip_dims=(220, 300), cropped=cropped, v=v)
  assert len(roi) == 4

def test_get_meta_roi_from_det():
  # No violations
  det = (100, 100, 180, 220)
  roi = get_meta_roi_from_det(det, clip_dims=(220, 300))
  assert roi == (84, 76, 196, 244)
  # Shift to left
  det = (10, 100, 90, 220)
  roi = get_meta_roi_from_det(det, clip_dims=(220, 300))
  assert roi == (0, 76, 106, 244)
  # Shift to right
  det = (140, 100, 220, 220)
  roi = get_meta_roi_from_det(det, clip_dims=(220, 300))
  assert roi == (124, 76, 220, 244)
  # Shift to top
  det = (100, 20, 180, 140)
  roi = get_meta_roi_from_det(det, clip_dims=(220, 300))
  assert roi == (84, 0, 196, 164)
  # Shift to bottom
  det = (100, 140, 180, 260)
  roi = get_meta_roi_from_det(det, clip_dims=(220, 300))
  assert roi == (84, 116, 196, 284)

def test_get_roi_from_det():
  det = (100, 100, 180, 220)
  assert get_roi_from_det(det, roi_method='face') == get_face_roi_from_det(det)
  assert get_roi_from_det(det, roi_method='forehead') == get_forehead_roi_from_det(det)
  assert get_roi_from_det(det, roi_method='upper_body', clip_dims=(220, 300)) == get_upper_body_roi_from_det(det, clip_dims=(220, 300))
  assert get_roi_from_det(det, roi_method='upper_body_cropped', clip_dims=(220, 300)) == get_upper_body_roi_from_det(det, clip_dims=(220, 300), cropped=True)
  assert get_roi_from_det(det, roi_method='meta', clip_dims=(220, 300)) == get_meta_roi_from_det(det, clip_dims=(220, 300))
  assert get_roi_from_det(det, roi_method=None) == det

@pytest.mark.parametrize("roi_method", ['face', 'forehead', 'upper_body', 'upper_body_cropped', 'meta', None])
@pytest.mark.parametrize("library", ['PIL', 'cv2', 'tf'])
def test_crop_resize_from_det(roi_method, library):
  # Uses prpy, only check shapes
  inputs = np.zeros((3, 300, 220, 3))
  inputs = inputs.astype(np.uint8)
  inputs_copy = inputs.copy()
  det = (100, 100, 180, 220)
  # face
  out = crop_resize_from_det(inputs, det=det, size=(36, 36), roi_method=roi_method, library=library, scale_algorithm='bicubic')
  assert out.shape == (3, 36, 36, 3)
  np.testing.assert_equal(inputs, inputs_copy) # No side effects
