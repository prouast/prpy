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
from typing import Optional, Union

import sys
sys.path.append('../prpy')
from prpy.numpy.core import standardize
from prpy.ffmpeg.utils import create_test_video_stream
from prpy.ffmpeg.readwrite import _ffmpeg_output_to_file, _ffmpeg_output_to_numpy

SAMPLE_FPS = 25.
SAMPLE_FRAMES = 250
SAMPLE_WIDTH = 320
SAMPLE_HEIGHT = 240

def _make_synthetic_ecg(
    fs: int,
    duration: float,
    hr_bpm: float,
    *,
    hr_trend_bpm_per_min: float = 0.0,
    missing_beat_at: Optional[int] = None,
    noise_std: float = 0.02,
    seed: Optional[Union[int, np.random.Generator]] = None,
  ) -> tuple[np.ndarray, np.ndarray]:
  """Synthetic ECG with optional HR drift and reproducible RNG."""
  rng = np.random.default_rng(seed)
  n = int(duration * fs)
  t = np.arange(n) / fs
  rr_start = 60.0 / hr_bpm
  rr_end = 60.0 / (hr_bpm + hr_trend_bpm_per_min * duration / 60)
  n_guess = int(duration / rr_start) + 4
  rr = np.linspace(rr_start, rr_end, n_guess)
  beats = np.cumsum(rr)
  beats = beats[beats < duration]
  if missing_beat_at is not None and missing_beat_at < len(beats):
    beats = np.delete(beats, missing_beat_at)
  sig = np.exp(-((t[:, None] - beats[None, :]) ** 2) / (2 * 0.015**2)).sum(axis=1)
  f = np.fft.rfftfreq(n, 1 / fs)
  pink = np.fft.irfft(rng.standard_normal(f.size) / np.maximum(f, 1e-6))
  sig += 0.02 * standardize(pink)
  sig = standardize(sig) + rng.normal(0, noise_std, n)
  return t, sig

def _make_synthetic_ppg(
    fs: int,
    duration: float,
    hr_bpm: float,
    *,
    hr_trend_bpm_per_min: float = 0.0,
    missing_beat_at: Optional[int] = None,
    noise_std: float = 0.02,
    seed: Optional[Union[int, np.random.Generator]] = None,
  ) -> tuple[np.ndarray, np.ndarray]:
  """Synthetic PPG with optional HR drift and reproducible RNG."""
  rng = np.random.default_rng(seed)
  n = int(duration * fs)
  t = np.arange(n) / fs
  rr_start = 60.0 / hr_bpm
  rr_end = 60.0 / (hr_bpm + hr_trend_bpm_per_min * duration / 60)
  rr_guess = int(duration / rr_start) + 4
  rr = np.linspace(rr_start, rr_end, rr_guess)
  beats = np.cumsum(rr)
  beats = beats[beats < duration]
  if missing_beat_at is not None and missing_beat_at < len(beats):
    beats = np.delete(beats, missing_beat_at)
  if len(beats) > 1:
    rr = np.diff(np.concatenate([beats, [beats[-1] + rr[-2]]]))
  else:
    rr = np.array([duration])
  def pulse_shape(phi: np.ndarray) -> np.ndarray:
    y = np.empty_like(phi)
    rise_frac = 0.2
    rise_mask = phi < rise_frac
    fall_mask = ~rise_mask
    if rise_mask.any():
      y[rise_mask] = np.sin(np.pi * phi[rise_mask] / (2 * rise_frac)) ** 3
    if fall_mask.any():
      x = (phi[fall_mask] - rise_frac) / (1 - rise_frac)
      decay = np.exp(-4.0 * x)
      notch = 0.15 * np.exp(-((phi[fall_mask] - 0.6) ** 2) / (2 * 0.01))
      y[fall_mask] = decay - notch
    return y
  sig = np.zeros(n)
  for i, bt in enumerate(beats):
    rr_i = rr[i] if i < len(rr) else rr[-1]
    start = int(bt * fs)
    stop = int(min((bt + rr_i) * fs, n))
    if stop <= start:
      continue
    phi = (t[start:stop] - bt) / rr_i
    sig[start:stop] = pulse_shape(phi)
  sig += 0.05 * np.sin(2 * np.pi * 0.15 * t)
  sig = (sig - sig.mean()) / (sig.std() + 1e-12)
  sig += rng.normal(0, noise_std, n)
  return t, sig

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

@pytest.fixture(scope='session')
def synthetic_ecg_dynamic():
  _, sig = _make_synthetic_ecg(
    fs=250,
    duration=30,
    hr_bpm=60.,
    hr_trend_bpm_per_min=30.0,
    missing_beat_at=8,
    seed=3
  )
  return sig

@pytest.fixture(scope='session')
def synthetic_ppg_dynamic():
  _, sig = _make_synthetic_ppg(
    fs=250,
    duration=30,
    hr_bpm=60.,
    hr_trend_bpm_per_min=30.0,
    missing_beat_at=8,
    seed=3
  )
  return sig

@pytest.fixture(scope='session')
def synthetic_ecg_static():
  _, sig = _make_synthetic_ecg(
    fs=250,
    duration=30,
    hr_bpm=60.,
    hr_trend_bpm_per_min=0.0,
    missing_beat_at=8,
    seed=3
  )
  return sig

@pytest.fixture(scope='session')
def synthetic_ppg_static():
  _, sig = _make_synthetic_ppg(
    fs=250,
    duration=30,
    hr_bpm=60.,
    hr_trend_bpm_per_min=0.0,
    missing_beat_at=8,
    seed=3
  )
  return sig

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
          