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
import struct
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

def _parse_ffprobe_csv_floats(raw: str) -> np.ndarray:
  """Parse ffprobe's `-of csv=p=0` stdout (one value per line, possibly with a
  trailing comma) into a float array."""
  cleaned = "\n".join(line.rstrip(",") for line in raw.splitlines())
  return np.fromstring(cleaned, dtype=float, sep="\n")

class _MP4IndexUnusable(Exception):
  pass

def _mp4_boxes(f, start: int, end: int):
  """Yield (type, payload_start, box_end) for the MP4 boxes in [start, end)."""
  pos = start
  while pos + 8 <= end:
    f.seek(pos)
    size, typ = struct.unpack('>I4s', f.read(8))
    header = 8
    if size == 1:
      size = struct.unpack('>Q', f.read(8))[0]
      header = 16
    elif size == 0:
      size = end - pos
    if size < header or pos + size > end:
      raise _MP4IndexUnusable(f"bad box size at {pos}")
    yield typ, pos + header, pos + size
    pos += size

def _mp4_child(f, start: int, end: int, name: bytes, required: bool = True):
  for typ, s, e in _mp4_boxes(f, start, end):
    if typ == name:
      return s, e
  if required:
    raise _MP4IndexUnusable(f"no {name.decode()} box")
  return None

def _mp4_full_box(f, start: int) -> Tuple[int, int]:
  """Seek to a full box payload, return (version, entry count)."""
  f.seek(start)
  version = f.read(4)[0]
  return version, struct.unpack('>I', f.read(4))[0]

def _mp4_table(f, n: int) -> np.ndarray:
  data = f.read(8 * n)
  if len(data) != 8 * n:
    raise _MP4IndexUnusable("truncated sample table")
  return np.frombuffer(data, dtype='>u4').reshape(-1, 2)

def _mp4_index_frame_timestamps(path: str) -> np.ndarray:
  """Read the frame timestamps of the first video track from an MP4/MOV index.

  Reads only the sample tables (stts, ctts, elst), not the media data.
  Raises _MP4IndexUnusable for anything this does not handle.
  """
  size = os.path.getsize(path)
  with open(path, 'rb') as f:
    if f.read(8)[4:8] != b'ftyp':
      raise _MP4IndexUnusable("not an MP4/MOV file")
    moov = _mp4_child(f, 0, size, b'moov')
    if _mp4_child(f, *moov, b'mvex', required=False):
      raise _MP4IndexUnusable("fragmented MP4")
    for typ, s, e in _mp4_boxes(f, *moov):
      if typ != b'trak':
        continue
      mdia = _mp4_child(f, s, e, b'mdia')
      f.seek(_mp4_child(f, *mdia, b'hdlr')[0] + 8)
      if f.read(4) != b'vide':
        continue
      f.seek(_mp4_child(f, *mdia, b'mdhd')[0])
      version = f.read(4)[0]
      f.read(16 if version == 1 else 8)
      timescale = struct.unpack('>I', f.read(4))[0]
      if timescale == 0:
        raise _MP4IndexUnusable("zero timescale")
      stbl = _mp4_child(f, *_mp4_child(f, *mdia, b'minf'), b'stbl')
      _, n = _mp4_full_box(f, _mp4_child(f, *stbl, b'stts')[0])
      stts = _mp4_table(f, n)
      dts = np.concatenate([[0], np.cumsum(np.repeat(stts[:, 1].astype(np.int64), stts[:, 0]))[:-1]])
      pts = dts
      ctts_box = _mp4_child(f, *stbl, b'ctts', required=False)
      if ctts_box:
        version, n = _mp4_full_box(f, ctts_box[0])
        ctts = _mp4_table(f, n)
        offsets = ctts[:, 1].view('>i4') if version == 1 else ctts[:, 1]
        offsets = np.repeat(offsets.astype(np.int64), ctts[:, 0])
        if offsets.size != dts.size:
          raise _MP4IndexUnusable("ctts and stts sample counts differ")
        pts = dts + offsets
      shift = 0
      edts = _mp4_child(f, s, e, b'edts', required=False)
      elst = _mp4_child(f, *edts, b'elst', required=False) if edts else None
      if elst:
        version, n = _mp4_full_box(f, elst[0])
        if n != 1:
          raise _MP4IndexUnusable(f"{n} edit list entries")
        _, shift = struct.unpack('>Qq' if version == 1 else '>Ii', f.read(16 if version == 1 else 8))
        if shift < 0:
          raise _MP4IndexUnusable("empty edit")
      if pts.size == 0:
        raise _MP4IndexUnusable("no samples")
      return np.round(np.sort(pts - shift) / timescale, 6)
  raise _MP4IndexUnusable("no video track")

def probe_video_frame_timestamps(path: str, sanity_check: bool = False) -> list:
  """Probe a video file for a best effort estimate of its frame timestamps.

  For MP4/MOV, reads the timestamps from the file's index (no media data read).
  Otherwise tries reading each packet's pts directly and sorting (no decode
  needed), and falls back to decoding every frame.

  Args:
    path: The path of the video.
    sanity_check: Whether to check result against ffprobe count
  Returns:
    List with the frame timestamps in seconds.
  """
  if not os.path.exists(path):
    raise FileNotFoundError(f"File {path} does not exist")

  timestamps = None
  try:
    timestamps = _mp4_index_frame_timestamps(path)
    if timestamps.size > 1 and np.sum(np.diff(timestamps) <= 0) > max(1, int(0.01 * timestamps.size)):
      logging.debug("MP4 index has too many tied timestamps; probing packets.")
      timestamps = None
  except (_MP4IndexUnusable, OSError, struct.error, ValueError) as e:
    logging.debug("MP4 index unusable (%s); probing packets.", e)
  if timestamps is None:
    try:
      proc = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time",
        "-of", "csv=p=0",
        path
      ], capture_output=True, text=True, check=True)
      lines = proc.stdout.splitlines()
      if lines and all(line.rstrip(",") for line in lines):
        candidate = np.sort(_parse_ffprobe_csv_floats(proc.stdout))
        # A handful of tied pts (e.g. a GOP-boundary rounding artifact) is fine
        # -- sorting already guarantees non-decreasing order, so the only real
        # question is whether too many packets are tied to trust the result.
        n_dupes = int(np.sum(np.diff(candidate) <= 0)) if candidate.size else 0
        if candidate.size == len(lines) and n_dupes <= max(1, int(0.01 * candidate.size)):
          timestamps = candidate
        else:
          logging.debug("Packet-level pts unusable (incomplete or too many duplicates); falling back.")
    except subprocess.CalledProcessError as e:
      logging.debug("Fast packet-level probe failed, falling back: %s", e)

  if timestamps is None:
    # Fallback: decode every frame to read its best-effort timestamp.
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
    timestamps = _parse_ffprobe_csv_floats(raw)

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
