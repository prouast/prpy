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
from prpy.numpy.image import probe_image_inputs

import math
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

def test_resample_bilinear_non_square_target_against_reference():
  height, width = 10, 20
  gradient = np.linspace(0, 255, width, dtype=np.uint8)
  broadcastable_gradient = gradient[np.newaxis, np.newaxis, :, np.newaxis]
  image_in = np.ascontiguousarray(np.broadcast_to(broadcastable_gradient, (1, height, width, 3)))
  image_in_copy = image_in.copy()
  target_size = (4, 8)
  actual_result = np.squeeze(resample_bilinear(im=image_in, size=target_size))
  expected_result = crop_slice_resize(
    inputs=image_in,
    target_size=target_size,
    library='PIL',
    scale_algorithm='bilinear'
  )
  expected_result = np.squeeze(expected_result, axis=0)
  np.testing.assert_allclose(actual_result, expected_result, atol=5)
  np.testing.assert_equal(image_in, image_in_copy)

def test_resample_bilinear_upsampling_correctness():
  # Create a 2x2 image with distinct corners
  im_in = np.array([
    [[255,   0,   0], [  0, 255,   0]],
    [[  0,   0, 255], [255, 255,   0]]
  ], dtype=np.uint8)
  im_in = im_in[np.newaxis, :, :, :] # Add batch dimension
  # Upsample to 3x3
  im_out = resample_bilinear(im=im_in, size=(3, 3))
  im_out = np.squeeze(im_out, axis=0)
  # Corners are now extrapolated values
  np.testing.assert_allclose(im_out[0, 0], [255, 0, 0], atol=1)
  np.testing.assert_allclose(im_out[0, 2], [0, 255, 0], atol=1)
  np.testing.assert_allclose(im_out[2, 0], [0, 0, 255], atol=1)
  np.testing.assert_allclose(im_out[2, 2], [255, 255, 0], atol=1)
  # Interpolated value between Red and Green
  np.testing.assert_allclose(im_out[0, 1], [127, 127,   0], atol=1)
  # The very center should be an equal mix of all four corners
  np.testing.assert_allclose(im_out[1, 1], [127, 127, 63], atol=1)

def test_resample_bilinear_downsampling_correctness():
  """
  Tests if downsampling a simple 4x1 gradient produces the correct middle value.
  """
  # Create a simple 4-pixel wide gradient
  im_in = np.array([[
    [0, 0, 0], [85, 85, 85], [170, 170, 170], [255, 255, 255]
  ]], dtype=np.uint8)
  im_in = im_in[np.newaxis, :, :, :] # Add batch dimension
  # Downsample to 2x1
  im_out = resample_bilinear(im=im_in, size=(1, 2))
  im_out = np.squeeze(im_out) # Remove all single dimensions
  # Expected result for 'half-pixel centers' resampling
  expected_result = np.array([
    [42, 42, 42], [212, 212, 212]
  ], dtype=np.uint8)
  np.testing.assert_allclose(im_out, expected_result, atol=1)

def test_resample_bilinear_segfault_memleak():
  test_video_ndarray = np.random.randint(0, 256, size=(138, 720, 1080, 3), dtype=np.uint8)
  _ = resample_bilinear(test_video_ndarray, 200)
  mem_0 = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
  for _ in range(10):
    test_video_ndarray = np.random.randint(0, 256, size=(138, 720, 1080, 3), dtype=np.uint8)
    _ = resample_bilinear(test_video_ndarray, 200)
    mem_1 = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    assert mem_1 - mem_0 < 1

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

def test_resample_box_perfect_averaging():
  """
  Tests box resampling where the target size is a perfect divisor of the source.
  """
  # Create a 4x4 image with four distinct 2x2 quadrants
  im_in = np.zeros((4, 4, 3), dtype=np.uint8)
  im_in[0:2, 0:2, :] = [10, 20, 30]
  im_in[0:2, 2:4, :] = [40, 50, 60]
  im_in[2:4, 0:2, :] = [70, 80, 90]
  im_in[2:4, 2:4, :] = [100, 110, 120]
  im_in = im_in[np.newaxis, :, :, :]
  # Downsample to 2x2
  im_out = resample_box(im=im_in, size=(2, 2))
  im_out = np.squeeze(im_out, axis=0)
  # Expected result: each output pixel is the exact average of its 2x2 block
  expected_result = np.array([
    [[10, 20, 30], [40, 50, 60]],
    [[70, 80, 90], [100, 110, 120]]
  ], dtype=np.uint8)
  np.testing.assert_array_equal(im_out, expected_result)

def test_resample_box_non_integer_ratio():
  # Create a 10x10 image
  im_in = np.arange(100, dtype=np.uint8).reshape(10, 10)
  # Make it a 3-channel image for the function
  im_in = np.stack([im_in, im_in, im_in], axis=-1)
  im_in = im_in[np.newaxis, :, :, :] # Add batch dimension
  # Manually calculate the expected value for the top-left (0,0) output pixel.
  # For a 10->3 resize, the first block covers indices [0,1,2] in both x and y.
  # Source box is im_in[0, 0:3, 0:3, 0]
  expected_top_left_pixel_val = np.mean(im_in[0, 0:3, 0:3, 0])
  # Downsample to 3x3
  im_out = resample_box(im=im_in, size=(3, 3))
  # Check the shape and the calculated pixel
  assert im_out.shape == (1, 3, 3, 3)
  np.testing.assert_allclose(im_out[0, 0, 0], expected_top_left_pixel_val, atol=1)

@pytest.mark.parametrize("resample_func", [resample_bilinear, resample_box])
def test_resampling_with_non_contiguous_input(resample_func):
  base_array = np.random.randint(0, 256, (1, 20, 20, 3), dtype=np.uint8)
  # Create a non-contiguous view by skipping every other pixel
  im_non_contiguous = base_array[:, ::2, ::2, :]
  assert im_non_contiguous.flags['C_CONTIGUOUS'] is False
  assert im_non_contiguous.shape == (1, 10, 10, 3)
  # Create a contiguous version for comparison
  im_contiguous = im_non_contiguous.copy()
  assert im_contiguous.flags['C_CONTIGUOUS'] is True
  # Resample both versions
  result_from_non_contiguous = resample_func(im=im_non_contiguous, size=(5, 5))
  result_from_contiguous = resample_func(im=im_contiguous, size=(5, 5))
  # The results should be identical
  np.testing.assert_array_equal(result_from_non_contiguous, result_from_contiguous)

@pytest.mark.parametrize("resample_func", [resample_bilinear, resample_box])
@pytest.mark.parametrize("shape", [(0, 10, 10, 3), (1, 0, 10, 3), (1, 10, 0, 3)])
def test_resampling_with_empty_input(resample_func, shape):
  im_in = np.zeros(shape, dtype=np.uint8)
  target_size = (5, 5)
  im_out = resample_func(im=im_in, size=target_size)
  expected_h = 0 if shape[1] == 0 else target_size[0]
  expected_w = 0 if shape[2] == 0 else target_size[1]
  expected_shape = (shape[0], expected_h, expected_w, shape[3])
  assert im_out.shape == expected_shape
  assert im_out.size == 0

@pytest.mark.parametrize("resample_func", [resample_bilinear, resample_box])
def test_resampling_invalid_dtype_raises_error(resample_func):
  im_in_float = np.zeros((1, 10, 10, 3), dtype=np.float32)
  with pytest.raises(ValueError, match="Input array must be .* and of type uint8"):
    resample_func(im=im_in_float, size=(5, 5))

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

def test_reduce_roi_non_contiguous_input():
  video = np.arange(2 * 10 * 10 * 3, dtype=np.uint8).reshape((2, 10, 10, 3))
  # Create a larger ROI array and then slice it to make it non-contiguous
  all_rois = np.array([
      [1, 1, 3, 3], # Corresponds to video frame 0
      [9, 9, 9, 9], # Dummy row
      [2, 2, 4, 4], # Corresponds to video frame 1
      [9, 9, 9, 9]  # Dummy row
  ], dtype=np.int64)
  roi_sliced = all_rois[::2] # This creates a non-contiguous view
  assert not roi_sliced.flags['C_CONTIGUOUS']
  # Calculate result with our implementation
  out = reduce_roi(video=video, roi=roi_sliced)
  # Calculate expected result using numpy
  exp_0 = np.mean(video[0, 1:3, 1:3, :], axis=(0, 1))
  exp_1 = np.mean(video[1, 2:4, 2:4, :], axis=(0, 1))
  exp = np.vstack([exp_0, exp_1])
  np.testing.assert_allclose(out, exp, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("in_mode", ["np", "str"])
@pytest.mark.parametrize("in_type", ["image", "video"])
def test_probe_image_inputs(sample_image_file, sample_image_data, sample_video_file, sample_video_data, sample_dims,
                            in_mode, in_type):
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
  shape, fps, issues = probe_image_inputs(inputs=inputs_in,
                                          fps=25. if in_mode == "np" and in_type == "video" else None)
  expected_shape = sample_dims if in_type == "video" else sample_dims[1:]
  expected_fps = 25. if in_type == "video" else None
  expected_issues = False
  assert shape == expected_shape
  assert fps == expected_fps
  assert issues == expected_issues

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
    expected_frames = math.ceil(sample_dims[0] / 5)
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
