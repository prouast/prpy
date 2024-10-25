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

import ffmpeg
from fractions import Fraction
import logging
from typing import Tuple
import os

def probe_video(
    path: str
  ) -> Tuple[float, int, int, int, str, float, int, bool]:
  """Probe a video file for metadata.

  Args:
    path: The path of the video.
  Returns:
    Tuple of
     - fps: The frame rate of the video
     - total_frames: The total number of frames
     - width: The width dimension of the video
     - height: The height dimension of the video
     - codec: The codec of the video
     - bitrate: The bitrate of the video
     - rotation: The rotation of the video
     - issues: Indicates if there are issues with the video
  """
  # Check if file exists
  assert isinstance(path, str)
  if not os.path.exists(path):
    raise FileNotFoundError("File {} does not exist".format(path))
  # ffprobe -show_streams -count_frames -pretty video.mp4
  try:
    probe = ffmpeg.probe(filename=path)
  except Exception as e:
    # The exception returned by `ffprobe` is in bytes
    logging.warning("Exception probing video: {}".format(e))
  else:
    # Check if the file contains video streams
    if 'streams' not in probe or not any(s['codec_type'] == 'video' for s in probe['streams']):
      raise ValueError("No video streams found")
    video_stream = next(
      (
        stream
        for stream in probe["streams"]
        if stream["codec_type"] == "video"
      ),
      None,
    )
    issues = False
    try:
      fps = float(Fraction(video_stream["avg_frame_rate"]))
    except Exception as e:
      logging.warning("Frame rate information missing")
      issues = True
      fps = None
    try:
      duration = float(video_stream['duration'])
    except Exception as e:
      duration = None
    try:
      total_frames = int(video_stream["nb_frames"])
    except Exception as e:
      issues = True
      if fps is not None and duration is not None:
        logging.warning("Number of frames missing. Inferring using duration and fps.")
        total_frames = int(duration*fps)
      else:
        logging.warning("Cannot infer number of total frames")
        total_frames = None
    width = video_stream["width"]
    height = video_stream["height"]
    codec = video_stream["codec_name"]
    try:
      bitrate = float(video_stream["bit_rate"])/1000.0
    except Exception as e:
      logging.warning("Bitrate information missing")
      bitrate = None
    rotation = 0
    if 'tags' in video_stream and 'rotate' in video_stream['tags']:
      # Regular
      rotation = int(video_stream['tags']['rotate'])
    elif 'side_data_list' in video_stream and 'rotation' in video_stream['side_data_list'][0]:
      # iPhone
      rotation = int(video_stream['side_data_list'][0]['rotation'])
    if not issues:
      # Check for other issues
      if total_frames is not None and duration is not None and fps is not None:
        expected_frames = int(duration * fps)
        if abs(total_frames - expected_frames) > 1:
          logging.warning("Mismatch between number of frames and duration / fps information")
          issues = True
    return fps, total_frames, width, height, codec, bitrate, rotation, issues
