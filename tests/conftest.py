###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import numpy as np
import os
import random as rand
import pytest

import sys
sys.path.append('../propy')
from propy.ffmpeg.utils import create_test_video_stream
from propy.ffmpeg.readwrite import _ffmpeg_output_to_file, _ffmpeg_output_to_numpy

SAMPLE_FPS = 25.
SAMPLE_FRAMES = 250
SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 240

@pytest.fixture
def random():
  rand.seed(0)
  np.random.seed(0)

@pytest.fixture
def sample_video_file(scope='session'):
  # Create sample stream
  stream = create_test_video_stream(10)
  # Write stream to h264 video file
  filename = 'test.mp4'
  _ffmpeg_output_to_file(stream, output_dir='', output_file=filename, overwrite=True)
  yield filename
  # After all tests finished
  os.remove(filename)

@pytest.fixture
def sample_video_data():
  # Create sample stream
  stream = create_test_video_stream(10)
  # Get np array for stream
  data = _ffmpeg_output_to_numpy(
    stream, target_fps=SAMPLE_FPS, target_n=SAMPLE_FRAMES,
    target_w=SAMPLE_WIDTH, target_h=SAMPLE_HEIGHT, pix_fmt='rgb24')
  return data
