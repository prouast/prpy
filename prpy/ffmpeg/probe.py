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

from fractions import Fraction
import logging
from typing import Tuple
import os
import json
import subprocess
import numpy as np

def probe_video(
    path: str,
    need_exact_framecount: bool = False
  ) -> Tuple[float, int, int, int, str, float, int, bool]:
  """Probe a video file for metadata.

  Args:
    path: The path to the video file.
    need_exact_framecount: If True and frame count info is missing,
                              re-run ffprobe with -count_frames for an accurate count.
                           If False, estimates frame count from duration * fps.
  Returns:
    A tuple containing:
      - fps: Frame rate (float)
      - total_frames: Total number of frames (int)
      - width: Video width (int)
      - height: Video height (int)
      - codec: Codec name (str)
      - bitrate: Bitrate in kb/s (float)
      - rotation: Rotation in degrees (int)
      - issues: True if any issues were encountered (bool)
  """
  if not isinstance(path, str):
    raise ValueError("Path must be a string")
  if not os.path.exists(path):
    raise FileNotFoundError(f"File {path} does not exist")

  def run_probe(count_frames: bool) -> dict:
    cmd = [
      "ffprobe",
      "-v", "error",
      "-show_streams",
      "-show_format",
      "-print_format", "json",
    ]
    if count_frames:
      cmd.insert(3, "-count_frames")
    cmd.append(path)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

  # First probe without -count_frames for speed.
  try:
    probe = run_probe(count_frames=False)
  except Exception as e:
    logging.warning(f"Exception probing video: {e}")
    raise e

  streams = probe.get("streams", [])
  video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
  if video_stream is None:
    raise ValueError("No video streams found")

  issues = False

  # Get frame rate.
  try:
    # avg_frame_rate is usually a string like "28.67/1"
    fps = float(Fraction(video_stream.get("avg_frame_rate", "0/0")))
    if fps == 0:
      raise ValueError
  except Exception:
    logging.warning("Frame rate information missing")
    issues = True
    fps = None

  # Duration: try stream then format metadata.
  try:
    duration = float(video_stream.get("duration", probe.get("format", {}).get("duration")))
  except Exception:
    duration = None

  # Get total frames.
  frames_str = video_stream.get("nb_frames") or video_stream.get("nb_read_frames")
  if frames_str is None:
    if need_exact_framecount:
      # Re-run ffprobe with -count_frames if frame count info is missing.
      try:
        logging.debug("Frame count info missing. Re-running ffprobe with -count_frames.")
        probe = run_probe(count_frames=True)
        streams = probe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        frames_str = video_stream.get("nb_frames") or video_stream.get("nb_read_frames")
      except Exception as e:
        logging.warning(f"Re-run with -count_frames failed: {e}.")
        frames_str = None
    else:
      logging.warning("Frame count info missing. Falling back to estimation using duration and fps.")

  try:
    total_frames = int(frames_str)
  except Exception:
    issues = True
    if fps is not None and duration is not None:
      logging.warning("Number of frames missing. Inferring using duration and fps.")
      total_frames = int(duration * fps)
    else:
      logging.warning("Cannot infer number of total frames")
      total_frames = None

  # Get dimensions and codec.
  width = video_stream.get("width")
  height = video_stream.get("height")
  codec = video_stream.get("codec_name")

  # Bitrate: try the stream first, then format metadata.
  try:
    bitrate = float(video_stream.get("bit_rate"))
    bitrate = bitrate / 1000.0  # Convert to kb/s.
  except Exception:
    if "bit_rate" in probe.get("format", {}):
      try:
        bitrate = float(probe["format"]["bit_rate"]) / 1000.0
        logging.debug("Bitrate fetched from format metadata.")
      except Exception:
        logging.warning("Bitrate information missing")
        bitrate = None
    else:
      logging.warning("Bitrate information missing")
      bitrate = None

  # Rotation: check tags and side_data_list.
  rotation = 0
  if "tags" in video_stream and "rotate" in video_stream["tags"]:
    try:
      rotation = int(video_stream["tags"]["rotate"])
    except Exception:
      rotation = 0
  elif "side_data_list" in video_stream and video_stream["side_data_list"]:
    for data in video_stream["side_data_list"]:
      if "rotation" in data:
        try:
          rotation = int(data["rotation"])
        except Exception:
          rotation = 0
        break

  # Check for mismatches between total_frames and duration * fps.
  if not issues and total_frames is not None and duration is not None and fps is not None:
    expected_frames = int(duration * fps)
    if abs(total_frames - expected_frames) > 1:
      logging.warning("Mismatch between total frames and duration/fps information")
      issues = True

  return fps, total_frames, width, height, codec, bitrate, rotation, issues

def probe_video_frame_timestamps(path: str, sanity_check: bool = False) -> list:
  """Probe a video file for a best effort estimate of its frame timestamps.

  Args:
    path: The path of the video.
    sanity_check: Whether to check result against ffprobe count
  Returns:
    List with the frame timestamps in seconds.
  """
  if not os.path.exists(path):
    raise FileNotFoundError(f"File {path} does not exist")

  # Build the ffprobe command to extract the best effort timestamp for each frame.
  ts_cmd = [
    "ffprobe", "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "frame=best_effort_timestamp_time",
    "-of", "csv=p=0",
    path
  ]
  try:
    proc = subprocess.run(ts_cmd, capture_output=True, text=True, check=True)
    raw = proc.stdout
  except subprocess.CalledProcessError as e:
    logging.error("ffprobe error: %s", e)
    return []
  # Clean trailing commas and process with numpy
  cleaned = "\n".join(line.rstrip(",") for line in raw.splitlines())
  timestamps = np.fromstring(cleaned, dtype=float, sep="\n")

  if sanity_check:
    # Grab the frame count
    count_cmd = [
      "ffprobe", "-v", "error",
      "-count_frames",
      "-select_streams", "v:0",
      "-show_entries", "stream=nb_read_frames",
      "-of", "default=nokey=1:noprint_wrappers=1",
      path
    ]
    try:
      proc = subprocess.run(count_cmd, capture_output=True, text=True, check=True)
      expected_frames = int(proc.stdout.strip())
      if timestamps.size != expected_frames:
        logging.warning(
          "Frame count mismatch: parsed %d timestamps but ffprobe reports %d frames",
          timestamps.size, expected_frames
        )
    except Exception:
      expected_frames = None
      logging.warning("Could not perform sanity check")
  return timestamps
