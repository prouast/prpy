###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import numpy as np
import random as rand
import pytest

import sys
sys.path.append('../propy')
from propy.ffmpeg.utils import create_test_video_stream
from propy.ffmpeg.readwrite import _ffmpeg_output_to_file, _ffmpeg_output_to_numpy

@pytest.fixture
def random():
  rand.seed(0)
  np.random.seed(0)

@pytest.fixture
def sample_video_file():
  # Create sample stream
  stream = create_test_video_stream(10)
  # Write stream to h264 video file
  filename = 'test.mp4'
  _ffmpeg_output_to_file(stream, output_dir='', output_file=filename, overwrite=True)
  return filename

@pytest.fixture
def sample_video_data():
  # Create sample stream
  stream = create_test_video_stream(10)
  # Get np array for stream
  data, _ = _ffmpeg_output_to_numpy(stream)
  return data
