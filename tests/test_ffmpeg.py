###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import math
import numpy as np
import os
import pytest

import sys
sys.path.append('../propy')

from propy.ffmpeg.probe import probe_video
from propy.ffmpeg.readwrite import read_video_from_path
from propy.ffmpeg.readwrite import write_video_from_path, write_video_from_numpy

SAMPLE_FPS = 25.
SAMPLE_FRAMES = 250
SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 240
SAMPLE_CHANNELS = 3
SAMPLE_CODEC = 'h264'
SAMPLE_BITRATE = 62.144
SAMPLE_ROTATION = 0

def test_probe_video(sample_video_file):
  out = probe_video(path=sample_video_file)
  assert out[0:7] == (SAMPLE_FPS, SAMPLE_FRAMES, SAMPLE_WIDTH, SAMPLE_HEIGHT, SAMPLE_CODEC, SAMPLE_BITRATE, SAMPLE_ROTATION)

# TODO: Try scalar instead tuple for scale
@pytest.mark.parametrize("target_fps", [25., 8.])
@pytest.mark.parametrize("crop", [None, (256, 94, 160, 120)])
@pytest.mark.parametrize("scale", [None, (40, 40)])
@pytest.mark.parametrize("preserve_aspect_ratio", [False, True])
@pytest.mark.parametrize("trim", [None, (124, 249)])
@pytest.mark.parametrize("scale_algorithm", ["bicubic", "bilinear", "area", "lanczos"])
def test_read_video_from_path(sample_video_file, target_fps, crop, scale, trim, preserve_aspect_ratio, scale_algorithm):
  frames, ds_factor = read_video_from_path(
      path=sample_video_file, target_fps=target_fps, crop=crop, scale=scale,
      trim=trim, preserve_aspect_ratio=preserve_aspect_ratio,
      dim_deltas=(1, 1, 1), scale_algorithm=scale_algorithm)
  cor_ds_factor = SAMPLE_FPS // target_fps
  cor_frames = SAMPLE_FRAMES if trim is None else SAMPLE_FRAMES - (trim[1] - trim[0])
  cor_frames = math.ceil(cor_frames / cor_ds_factor)
  cor_height = SAMPLE_HEIGHT if crop is None else crop[3]
  cor_width = SAMPLE_WIDTH if crop is None else crop[2]
  if scale is not None and preserve_aspect_ratio:
    scale_ratio = max(scale) / max(cor_height, cor_width)
    cor_height = int(scale_ratio * cor_height)
    cor_width = int(scale_ratio * cor_width)
  else:
    cor_height = cor_height if scale is None else scale[1]
    cor_width = cor_width if scale is None else scale[0]
  assert ds_factor == cor_ds_factor
  assert frames.shape == (cor_frames, cor_height, cor_width, SAMPLE_CHANNELS)

def test_write_video_from_path(sample_video_file):
  test_filename = "test_out.mp4"
  write_video_from_path(
    sample_video_file, output_dir="", output_file=test_filename, target_fps=None,
    crop=None, scale=None, trim=None, crf=0, preserve_aspect_ratio=False, overwrite=True)
  frames_orig, _ = read_video_from_path(path=sample_video_file)
  frames_test, _ = read_video_from_path(path=test_filename)
  np.testing.assert_allclose(frames_test, frames_orig, rtol=1e-4)
  os.remove(test_filename)

def test_write_video_from_numpy(sample_video_data):
  test_filename = "test_out.mp4"
  write_video_from_numpy(
    sample_video_data, fps=SAMPLE_FPS, pix_fmt='rgb24', output_dir="",
    output_file=test_filename, out_pix_fmt='rgb24', crf=0, overwrite=True)
  frames_test, _ = read_video_from_path(path=test_filename, pix_fmt='rgb24')
  np.testing.assert_allclose(frames_test, sample_video_data, rtol=1e-4, atol=2)
  os.remove(test_filename)
  