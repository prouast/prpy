###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import sys
sys.path.append('../propy')

from propy.ffmpeg.probe import probe_video
from propy.ffmpeg.readwrite import read_video_from_path

import math
import numpy as np
import pytest

SAMPLE_FPS = 25.
SAMPLE_FRAMES = 250
SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 240
SAMPLE_CHANNELS = 3
SAMPLE_CODEC = 'h264'

def test_probe_video(sample_video_file):
  out = probe_video(path=sample_video_file)
  assert out[0:5] == (SAMPLE_FPS, SAMPLE_FRAMES, SAMPLE_WIDTH, SAMPLE_HEIGHT, SAMPLE_CODEC)

@pytest.mark.parametrize("target_fps", [25., 8.])
@pytest.mark.parametrize("crop", [None, (256, 94, 160, 120)])
@pytest.mark.parametrize("scale", [None, (40, 40)])
@pytest.mark.parametrize("preserve_aspect_ratio", [False, True])
@pytest.mark.parametrize("trim", [None, (124, 249)])
def test_read_video_from_path(sample_video_file, target_fps, crop, scale, trim, preserve_aspect_ratio):
  frames, ds_factor = read_video_from_path(
      path=sample_video_file, target_fps=target_fps, crop=crop, scale=scale,
      trim=trim, preserve_aspect_ratio=preserve_aspect_ratio)
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

