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

import math
import numpy as np
import os
import pytest

import sys
sys.path.append('../prpy')

from prpy.ffmpeg.probe import probe_video
from prpy.ffmpeg.readwrite import read_video_from_path
from prpy.ffmpeg.readwrite import write_video_from_path, write_video_from_numpy

SAMPLE_FPS = 25.
SAMPLE_FRAMES = 250
SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 240
SAMPLE_CHANNELS = 3
SAMPLE_CODEC = 'h264'
SAMPLE_BITRATE = 62.144
SAMPLE_ROTATION = 0
SAMPLE_ISSUES = False

def test_probe_video(sample_video_file):
  out = probe_video(path=sample_video_file)
  assert out[0:8] == (SAMPLE_FPS, SAMPLE_FRAMES, SAMPLE_WIDTH, SAMPLE_HEIGHT, SAMPLE_CODEC, SAMPLE_BITRATE, SAMPLE_ROTATION, SAMPLE_ISSUES)

@pytest.mark.parametrize("target_fps", [25., 8.])
@pytest.mark.parametrize("crop", [None, (256, 94, 416, 214)])
@pytest.mark.parametrize("scale", [None, 30, (40, 40)])
@pytest.mark.parametrize("preserve_aspect_ratio", [False, True])
@pytest.mark.parametrize("trim", [None, (124, 249)])
@pytest.mark.parametrize("scale_algorithm", ["bicubic", "bilinear"])
def test_read_video_from_path(sample_video_file, target_fps, crop, scale, trim, preserve_aspect_ratio, scale_algorithm):
  frames, ds_factor = read_video_from_path(
    path=sample_video_file, target_fps=target_fps, crop=crop, scale=scale,
    trim=trim, preserve_aspect_ratio=preserve_aspect_ratio, crf=None,
    dim_deltas=(0, 0, 0), scale_algorithm=scale_algorithm)
  cor_ds_factor = SAMPLE_FPS // target_fps
  cor_frames = SAMPLE_FRAMES if trim is None else SAMPLE_FRAMES - (trim[1] - trim[0])
  cor_frames = math.ceil(cor_frames / cor_ds_factor)
  cor_height = SAMPLE_HEIGHT if crop is None else crop[3]-crop[1]
  cor_width = SAMPLE_WIDTH if crop is None else crop[2]-crop[0]
  if isinstance(scale, int): scale = (scale, scale) 
  if scale is not None and preserve_aspect_ratio:
    scale_ratio = max(scale) / max(cor_height, cor_width)
    cor_height = int(scale_ratio * cor_height)
    cor_width = int(scale_ratio * cor_width)
  else:
    cor_height = cor_height if scale is None else scale[1]
    cor_width = cor_width if scale is None else scale[0]
  assert ds_factor == cor_ds_factor
  assert frames.shape == (cor_frames, cor_height, cor_width, SAMPLE_CHANNELS)

def test_read_video_from_path_uneven_crop(sample_video_file, caplog):
  frames, ds_factor = read_video_from_path(
    path=sample_video_file, target_fps=25., crop=(256, 94, 417, 214), scale=None, crf=None,
    trim=None, preserve_aspect_ratio=False, dim_deltas=(0, 0, 0), scale_algorithm='bicubic')
  cor_ds_factor = SAMPLE_FPS // 25.
  cor_frames = math.ceil(SAMPLE_FRAMES / cor_ds_factor)
  assert ds_factor == cor_ds_factor
  assert frames.shape == (cor_frames, 120, 160, SAMPLE_CHANNELS)
  assert "Reducing uneven crop width from 161 to 160 to make operation possible." in caplog.text

def test_read_video_from_path_uneven_crop_crf_scale(sample_video_file, caplog):
  frames, ds_factor = read_video_from_path(
    path=sample_video_file, target_fps=25., crop=(256, 94, 417, 214), scale=40,
    crf=12, order='crf_scale', trim=None, preserve_aspect_ratio=False, dim_deltas=(0, 0, 0),
    scale_algorithm='bicubic')
  cor_ds_factor = SAMPLE_FPS // 25.
  cor_frames = math.ceil(SAMPLE_FRAMES / cor_ds_factor)
  assert ds_factor == cor_ds_factor
  assert frames.shape == (cor_frames, 40, 40, SAMPLE_CHANNELS)
  assert "Reducing uneven crop width from 161 to 160 to make operation possible." in caplog.text

def test_read_video_from_path_uneven_crop_scale_crf(sample_video_file):
  frames, ds_factor = read_video_from_path(
    path=sample_video_file, target_fps=25., crop=(256, 94, 417, 214), scale=40,
    crf=12, order='scale_crf', trim=None, preserve_aspect_ratio=False, dim_deltas=(0, 0, 0),
    scale_algorithm='bicubic')
  cor_ds_factor = SAMPLE_FPS // 25.
  cor_frames = math.ceil(SAMPLE_FRAMES / cor_ds_factor)
  assert ds_factor == cor_ds_factor
  assert frames.shape == (cor_frames, 40, 40, SAMPLE_CHANNELS)

def test_write_video_from_path(temp_dir, sample_video_file):
  test_filename = "test_out.mp4"
  write_video_from_path(
    sample_video_file, output_dir=temp_dir, output_file=test_filename, target_fps=None,
    crop=(40, 60, 140, 200), scale=None, trim=None, crf=0, preserve_aspect_ratio=False,
    overwrite=True, codec='h264')
  frames_orig, _ = read_video_from_path(path=sample_video_file)
  frames_test, _ = read_video_from_path(path=os.path.join(temp_dir, test_filename))
  np.testing.assert_allclose(frames_test, frames_orig[:,60:200,40:140], rtol=1e-4)

def test_write_video_from_path_uneven_crop(temp_dir, sample_video_file, caplog):
  test_filename = "test_out.mp4"
  write_video_from_path(
    sample_video_file, output_dir=temp_dir, output_file=test_filename, target_fps=None,
    crop=(40, 60, 140, 201), scale=None, trim=None, crf=0, preserve_aspect_ratio=False,
    overwrite=True, codec='h264')
  frames_orig, _ = read_video_from_path(path=sample_video_file)
  frames_test, _ = read_video_from_path(path=os.path.join(temp_dir, test_filename))
  np.testing.assert_allclose(frames_test, frames_orig[:,60:200,40:140], rtol=1e-4)
  assert "Reducing uneven crop height from 141 to 140 to make operation possible." in caplog.text

def test_write_video_from_path_uneven_scale(temp_dir, sample_video_file, caplog):
  test_filename = "test_out.mp4"
  with pytest.raises(ValueError, match="Cannot use this scale"):
    write_video_from_path(
      sample_video_file, output_dir=temp_dir, output_file=test_filename, target_fps=None,
      crop=None, scale=41, trim=None, crf=0, preserve_aspect_ratio=False,
      overwrite=True, codec='h264')

def test_write_video_from_numpy(temp_dir, sample_video_data):
  test_filename = "test_out.mp4"
  sample_video_data_copy = sample_video_data.copy()
  write_video_from_numpy(
    sample_video_data, fps=SAMPLE_FPS, pix_fmt='rgb24', output_dir=temp_dir,
    output_file=test_filename, out_pix_fmt='rgb24', crf=0, overwrite=True)
  frames_test, _ = read_video_from_path(path=os.path.join(temp_dir, test_filename), pix_fmt='rgb24')
  np.testing.assert_allclose(frames_test, sample_video_data, rtol=1e-4, atol=2)
  np.testing.assert_equal(sample_video_data, sample_video_data_copy) # No side effects

def test_write_video_from_numpy_uneven_dims(temp_dir, sample_video_data):
  test_filename = "test_out.mp4"
  with pytest.raises(ValueError, match="H264 requires both height and width of the video to be even numbers"):
    write_video_from_numpy(
    sample_video_data[:,10:15,12:19], fps=SAMPLE_FPS, pix_fmt='rgb24', output_dir=temp_dir,
    output_file=test_filename, out_pix_fmt='rgb24', crf=0, overwrite=True)

def test_read_video_from_path_trim(sample_video_file, sample_video_data):
  # Indices to slice - end is excluded, so 2 frames
  # Sample video clock ticks over from first to second frame
  start_idx = 24
  end_idx = 26
  frames_sliced_directly = sample_video_data[start_idx:end_idx]
  frames_trim_ffmpeg, _ = read_video_from_path(sample_video_file, pix_fmt='rgb24',
                                               trim=(start_idx, end_idx))
  # Can't assert equality because there are some encoding artifacts.
  # Verified that if the trim timing was off, the np.mean would be about 12.0
  assert np.mean(frames_sliced_directly-frames_trim_ffmpeg) < 10.0
