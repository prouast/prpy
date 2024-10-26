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
from PIL import Image
import os
import pytest
import random as rand
import tempfile

import sys
sys.path.append('../prpy')
from prpy.ffmpeg.utils import create_test_video_stream
from prpy.ffmpeg.readwrite import _ffmpeg_output_to_file, _ffmpeg_output_to_numpy

SAMPLE_FPS = 25.
SAMPLE_FRAMES = 250
SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 240

@pytest.fixture(scope='session')
def temp_dir():
  with tempfile.TemporaryDirectory() as temp:
    yield temp

@pytest.fixture
def random():
  rand.seed(0)
  np.random.seed(0)

@pytest.fixture(scope='session')
def sample_video_file(temp_dir):
  # Create sample stream
  stream = create_test_video_stream(10)
  # Write stream to h264 video file
  filename = 'test.mp4'
  _ffmpeg_output_to_file(stream, output_dir=temp_dir, output_file=filename, overwrite=True)
  yield os.path.join(temp_dir, filename)

@pytest.fixture(scope='session')
def sample_video_data():
  # Create sample stream
  stream = create_test_video_stream(10)
  # Get np array for stream
  data = _ffmpeg_output_to_numpy(
    stream, r=0, fps=SAMPLE_FPS, n=SAMPLE_FRAMES, w=SAMPLE_WIDTH, h=SAMPLE_HEIGHT,
    pix_fmt='rgb24')
  return data

@pytest.fixture(scope='session')
def sample_image_file(temp_dir):
  # Create sample image
  random_image_data = np.random.randint(0, 256, (SAMPLE_HEIGHT, SAMPLE_WIDTH, 3), dtype=np.uint8)
  # Convert the array to an image
  image = Image.fromarray(random_image_data)
  # Save as a JPEG file
  filepath = os.path.join(temp_dir, "image.jpg")
  image.save(filepath, "JPEG")
  return filepath

@pytest.fixture(scope='session')
def sample_image_data():
  # Create sample image
  random_image_data = np.random.randint(0, 256, (SAMPLE_HEIGHT, SAMPLE_WIDTH, 3), dtype=np.uint8)
  return random_image_data

@pytest.fixture(scope='session')
def sample_dims():
  return SAMPLE_FRAMES, SAMPLE_HEIGHT, SAMPLE_WIDTH, 3

def pytest_collection_modifyitems(config, items):
  """Add marker for tests which parametrize cv2 or tf"""
  for item in items:
    if hasattr(item, 'callspec'):
      params = item.callspec.params
      for param in params.values():
        if param is not None and isinstance(param, tuple) or isinstance(param, str):
          if isinstance(param, str) and param in ['tf', 'cv2'] or \
             isinstance(param, tuple) and 'tf' in param or 'cv2' in param:
            item.add_marker(pytest.mark.skip_parametrize_tf_cv2)
          