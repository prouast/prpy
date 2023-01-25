###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, January 2023                #
###############################################################################

import numpy as np
import pytest

import sys
sys.path.append('../propy')

from propy.numpy.face import get_face_roi_from_det, get_forehead_roi_from_det, get_meta_roi_from_det
from propy.numpy.face import get_upper_body_roi_from_det, get_roi_from_det, crop_resize_from_det

def test_get_face_roi_from_det():
  det = (100, 100, 180, 220)
  roi = get_face_roi_from_det(det)
  assert roi == (116, 112, 164, 208)

def test_get_forehead_roi_from_det():
  det = (100, 100, 180, 220)
  roi = get_forehead_roi_from_det(det)
  assert roi == (128, 118, 152, 130)

@pytest.mark.parametrize("cropped", [True, False])
def test_get_upper_body_roi_from_det(cropped):
  # No violations
  det = (100, 100, 180, 220)
  roi = get_upper_body_roi_from_det(det, clip_dims=(220, 300), cropped=cropped)
  assert roi == (86, 82, 194, 256) if cropped else roi == (80, 76, 200, 268)
  # Shift to left
  det = (10, 100, 90, 220)
  roi = get_upper_body_roi_from_det(det, clip_dims=(220, 300), cropped=cropped)
  assert roi == (0, 82, 104, 256) if cropped else roi == (0, 76, 110, 268)
  # Shift to right
  det = (140, 100, 220, 220)
  roi = get_upper_body_roi_from_det(det, clip_dims=(220, 300), cropped=cropped)
  assert roi == (126, 82, 220, 256) if cropped else roi == (120, 76, 220, 268)
  # Shift to top
  det = (100, 20, 180, 140)
  roi = get_upper_body_roi_from_det(det, clip_dims=(220, 300), cropped=cropped)
  assert roi == (86, 2, 194, 176) if cropped else roi == (80, 0, 200, 188)
  # Shift to bottom
  det = (100, 140, 180, 260)
  roi = get_upper_body_roi_from_det(det, clip_dims=(220, 300), cropped=cropped)
  assert roi == (86, 122, 194, 296) if cropped else roi == (80, 116, 200, 300)

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
  assert get_roi_from_det(det, roi_method='meta', clip_dims=(220, 300)) == get_meta_roi_from_det(det, clip_dims=(220, 300))
  assert get_roi_from_det(det, roi_method=None) == det

@pytest.mark.parametrize("roi_method", ['face', 'forehead', 'upper_body', 'upper_body_cropped', 'meta', None])
@pytest.mark.parametrize("method", ['PIL', 'cv2', 'tf'])
def test_crop_resize_from_det(roi_method, method):
  # Uses propy, only check shapes
  inputs = np.zeros((3, 300, 220, 3))
  inputs = inputs.astype(np.uint8)
  det = (100, 100, 180, 220)
  # face
  out = crop_resize_from_det(inputs, det=det, size=(36, 36), roi_method=roi_method, method=method)
  assert out.shape == (3, 36, 36, 3)