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

from prpy.numpy.image import crop_slice_resize
from prpy.numpy.image import resample_bilinear, resample_box
from prpy.numpy.image import reduce_roi
from prpy.numpy.image import parse_image_inputs

import os
import numpy as np
import pytest
import psutil

@pytest.mark.parametrize("target_size", [6, 3, (6, 12)])
@pytest.mark.parametrize("n_frames", [None, 3])
@pytest.mark.parametrize("roi", [(2, 2, 7, 7), [2, 2, 7, 7], None])
@pytest.mark.parametrize("target_idxs", [None, [0, 2], (0, 2), np.asarray([0, 2])])
@pytest.mark.parametrize("preserve_aspect_ratio", [True, False])
@pytest.mark.parametrize("lib_algo", [("cv2", "bicubic"), ("cv2", "bilinear"),
                                      ("tf", "bicubic"), ("tf", "bilinear"),
                                      ("PIL", "bicubic"), ("PIL", "bilinear"),
                                      ("prpy", "bilinear")])
@pytest.mark.parametrize("keepdims", [True, False])
def test_crop_slice_resize(target_size, n_frames, roi, target_idxs, preserve_aspect_ratio, lib_algo, keepdims):
  if (n_frames is None and target_idxs is not None):
    pytest.skip("Skip because parameter combination does not work")
  library, scale_algorithm = lib_algo
  if n_frames is None:
    images_in = np.random.randint(size=(8, 12, 3), low=0, high=255)
  else:
    images_in = np.random.randint(size=(n_frames, 8, 12, 3), low=0, high=255)
  images_in = images_in.astype(np.uint8)
  images_in_copy = images_in.copy()
  images_out = crop_slice_resize(
    inputs=images_in, target_size=target_size, roi=roi, target_idxs=target_idxs,
    library=library, preserve_aspect_ratio=preserve_aspect_ratio,
    keepdims=keepdims, scale_algorithm=scale_algorithm)
  expected_frames = len(target_idxs) if target_idxs is not None else n_frames
  if expected_frames == 1 or expected_frames is None:
    expected_frames = 1 if keepdims else None
  if isinstance(target_size, int):
    if preserve_aspect_ratio is True and roi is None:
      expected_shape = (int(target_size*8./12), target_size, 3)
    else:
      expected_shape = (target_size, target_size, 3)
  else:
    if preserve_aspect_ratio is True and roi is None:
      expected_shape = (target_size[0], int(target_size[0]*12./8), 3)
    elif preserve_aspect_ratio is True and roi is not None:
      expected_shape = (target_size[0], target_size[0], 3)
    else:
      expected_shape = (target_size[0], target_size[1], 3)
  expected_shape = (expected_frames,) + expected_shape if expected_frames is not None else expected_shape
  assert images_out.shape == expected_shape
  if library == 'tf':
    import tensorflow as tf
    assert tf.is_tensor(images_out)
  else:
    assert isinstance(images_out, np.ndarray)
  # No side effects
  np.testing.assert_equal(images_in, images_in_copy)

def test_crop_slice_resize_retinaface():
  images_in = np.random.uniform(size=(480, 640, 3), low=0, high=255)  
  images_in = images_in.astype(np.uint8)
  images_in_copy = images_in.copy()
  images_out = crop_slice_resize(
    inputs=images_in, target_size=224, roi=(0, 0, 480, 640), target_idxs=None,
    library='PIL', preserve_aspect_ratio=True, keepdims=True, scale_algorithm='bicubic')
  assert images_out.shape == (1, 224, 224, 3)
  np.testing.assert_equal(images_in, images_in_copy)

@pytest.mark.parametrize("n_frames", [1, 3])
@pytest.mark.parametrize("size", [4, 12, (12, 16)])
def test_resample_bilinear(n_frames, size):
  # Note: Only tests for correct shape, not for correct pixel vals
  im = np.random.uniform(size=(n_frames, 8, 12, 3), low=0, high=255).astype(np.uint8)
  im_copy = im.copy()
  out = resample_bilinear(im=im, size=size)
  if isinstance(size, int): size = (size, size)
  assert out.shape == (n_frames, size[0], size[1], 3)
  np.testing.assert_equal(im, im_copy)

def test_resample_bilinear_segfault_memleak():
  test_video_ndarray = np.random.randint(0, 256, size=(138, 720, 1080, 3), dtype=np.uint8)
  _ = resample_bilinear(test_video_ndarray, 200)
  mem_0 = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
  for _ in range(10):
    test_video_ndarray = np.random.randint(0, 256, size=(138, 720, 1080, 3), dtype=np.uint8)
    _ = resample_bilinear(test_video_ndarray, 200)
    mem_1 = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    assert mem_1 - mem_0 < 1

@pytest.mark.parametrize("n_frames", [1, 3])
@pytest.mark.parametrize("size", [4, (2, 3)])
def test_resample_box(n_frames, size):
  # Note: Only tests for correct shape, not for correct pixel vals
  im = np.random.uniform(size=(n_frames, 8, 12, 3), low=0, high=255).astype(np.uint8)
  im_copy = im.copy()
  out = resample_box(im=im, size=size)
  if isinstance(size, int): size = (size, size)
  assert out.shape == (n_frames, size[0], size[1], 3)
  np.testing.assert_equal(im, im_copy)

def test_resample_box_segfault_memleak():
  test_video_ndarray = np.random.randint(0, 256, size=(138, 720, 1080, 3), dtype=np.uint8)
  _ = resample_box(test_video_ndarray, 200)
  mem_0 = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
  for _ in range(10):
    test_video_ndarray = np.random.randint(0, 256, size=(138, 720, 1080, 3), dtype=np.uint8)
    _ = resample_box(test_video_ndarray, 200)
    mem_1 = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    assert mem_1 - mem_0 < 1

def test_resample_box_only_downsampling():
  im = np.random.uniform(size=(3, 8, 12, 3), low=0, high=255).astype(np.uint8)
  im_copy = im.copy()
  with pytest.raises(Exception):
    _ = resample_box(im=im, size=5)
  np.testing.assert_equal(im, im_copy)

@pytest.mark.parametrize("scenario", [(1, [[2, 3, 4, 7]]),
                                      (3, [[3, 4, 7, 12], [3, 5, 7, 12], [3, 5, 6, 10]])])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_reduce_roi(scenario, dtype):
  n_frames = scenario[0]
  roi = np.asarray(scenario[1]).astype(dtype)
  video = np.random.uniform(size=(n_frames, 12, 8, 3), low=0, high=255).astype(np.uint8)
  video_copy = video.copy()
  out = reduce_roi(video=video, roi=roi)
  exp = np.asarray([np.mean(video[i, roi[i,1]:roi[i,3], roi[i,0]:roi[i,2]], axis=(0,1)) for i in range(n_frames)])
  assert out.shape == (n_frames, 3)
  np.testing.assert_allclose(
    out,
    exp,
    atol=1e-4, rtol=1e-4)
  np.testing.assert_equal(video, video_copy)

@pytest.mark.parametrize("in_mode_lib", [("np", "cv2"), ("np", "tf"), ("np", "PIL"), ("np", "prpy"), ("str", "prpy")])
@pytest.mark.parametrize("roi", [None, (100, 50, 220, 200)])
@pytest.mark.parametrize("scaling", [(None, False), ((40, 40), False), (40, True)])
@pytest.mark.parametrize("temporal", [("image", None, None, None, False),
                                      ("image", None, None, None, True),
                                      ("video", None, None, None, True),
                                      ("video", (10, 110), None, None, True),
                                      ("video", None, 5., None, True),
                                      ("video", None, None, 5, True),
                                      ("video", (10, 110), 5., None, True),
                                      ("video", (10, 110), None, 5, True)])
def test_parse_image_inputs(sample_image_file, sample_image_data, sample_video_file, sample_video_data, sample_dims,
                            in_mode_lib, roi, scaling, temporal):
  in_mode, library = in_mode_lib
  in_type, trim, target_fps, ds_factor, videodims = temporal
  target_size, preserve_aspect_ratio = scaling
  scale_algorithm = "bilinear"
  fps = 25. if in_mode == 'np' and in_type == 'video' else None
  if in_mode == "np":
    if in_type == "image":
      inputs_in = sample_image_data
    else:
      inputs_in = sample_video_data
  else:
    if in_type == "image":
      inputs_in = sample_image_file
    else:
      inputs_in = sample_video_file  
  parsed, fps_in, shape_in, ds_factor_used, idxs = parse_image_inputs(
    inputs=inputs_in, fps=fps, roi=roi, target_size=target_size, target_fps=target_fps,
    ds_factor=ds_factor, preserve_aspect_ratio=preserve_aspect_ratio,
    library=library, scale_algorithm=scale_algorithm, trim=trim,
    allow_image=True, videodims=videodims)
  expected_fps_in = 25. if in_type == 'video' else None
  expected_shape_in = sample_dims if in_type == 'video' else ((1,) + sample_dims[1:] if videodims else sample_dims[1:])
  if in_type == "image":
    expected_ds_factor = None
  elif ds_factor is None and target_fps is None:
    expected_ds_factor = 1
  elif ds_factor is not None:
    expected_ds_factor = ds_factor
  else:
    expected_ds_factor = int(target_fps)
  if in_type == "image":
    expected_frames = 1
    expected_idxs = [0]
  elif target_fps is None and ds_factor is None and trim is None:
    expected_frames = sample_dims[0]
    expected_idxs = list(range(sample_dims[0]))
  elif (target_fps is not None or ds_factor is not None) and trim is None:
    expected_frames = sample_dims[0] // 5
    expected_idxs = list(range(0, sample_dims[0], 5))
  elif target_fps is None and ds_factor is None and trim is not None:
    expected_frames = trim[1] - trim[0]
    expected_idxs = list(range(trim[0], trim[1]))
  else:
    assert trim is not None and (target_fps is not None or ds_factor is not None)
    expected_frames = (trim[1] - trim[0]) // 5
    expected_idxs = list(range(trim[0], trim[1], 5))
  target_size = (target_size, target_size) if isinstance(target_size, int) else target_size
  if roi is None and target_size is None:
    expected_height = sample_dims[1]
    expected_width = sample_dims[2]
  elif roi is not None and target_size is None and not preserve_aspect_ratio:
    roi_height = roi[3]-roi[1]
    roi_width = roi[2]-roi[0]
    expected_height = roi_height
    expected_width = roi_width
  elif roi is not None and target_size is None and preserve_aspect_ratio:
    roi_height = roi[3]-roi[1]
    roi_width = roi[2]-roi[0]
    expected_height = roi_height if roi_height > roi_width else int(roi_width * min(sample_dims[1], sample_dims[2])/max(sample_dims[1], sample_dims[2]))
    expected_width = roi_width if roi_width >= roi_height else int(roi_height * min(sample_dims[1], sample_dims[2])/max(sample_dims[1], sample_dims[2]))
  elif roi is None and target_size is not None and not preserve_aspect_ratio:
    expected_height = target_size[0]
    expected_width = target_size[1]
  elif roi is None and target_size is not None and preserve_aspect_ratio:
    expected_height = target_size[0] if target_size[0] > target_size[1] else int(target_size[1] * min(sample_dims[1], sample_dims[2])/max(sample_dims[1], sample_dims[2]))
    expected_width = target_size[1] if target_size[1] >= target_size[0] else int(target_size[0] * min(sample_dims[1], sample_dims[2])/max(sample_dims[1], sample_dims[2]))    
  elif target_size is not None and roi is not None and not preserve_aspect_ratio:
    expected_height = target_size[0]
    expected_width = target_size[1]
  else: # target_size is not None and roi is not None and preserve_aspect_ratio:
    roi_height = roi[3]-roi[1]
    roi_width = roi[2]-roi[0]
    expected_height = target_size[0] if roi_height > roi_width else int(target_size[1] * min(roi_height, roi_width)/max(roi_height, roi_width))
    expected_width = target_size[1] if roi_width >= roi_height else int(target_size[0] * min(roi_height, roi_width)/max(roi_height, roi_width))
  assert parsed.shape == ((expected_height, expected_width, 3) if (not videodims and in_type=="image") else (expected_frames, expected_height, expected_width, 3))
  assert fps_in == expected_fps_in
  assert shape_in == expected_shape_in
  assert ds_factor_used == expected_ds_factor
  assert idxs == expected_idxs

@pytest.mark.parametrize("in_mode_lib", [("np", "prpy"), ("str", "prpy")])
@pytest.mark.parametrize("temporal", [("image", None, None, None, False),
                                      ("image", None, None, None, True)])
def test_parse_image_inputs_error(sample_image_file, sample_image_data, sample_dims, in_mode_lib, temporal):
  in_mode, library = in_mode_lib
  _, trim, target_fps, ds_factor, videodims = temporal
  if in_mode == "np":
    inputs_in = sample_image_data
  else:
    inputs_in = sample_image_file
  with pytest.raises(Exception):
    parse_image_inputs(
      inputs=inputs_in, target_fps=target_fps,
      ds_factor=ds_factor, library=library, trim=trim, allow_image=False, videodims=videodims)
