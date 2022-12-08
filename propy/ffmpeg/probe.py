###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, September 2021              #
###############################################################################

import ffmpeg
from fractions import Fraction
import json
import logging
import subprocess

from propy.constants import FFPROBE_OK, FFPROBE_INFO, FFPROBE_CORR

def probe_video(path):
  # ffprobe -show_streams -count_frames -pretty video.mp4
  try:
    probe = ffmpeg.probe(filename=path)
  except Exception as e:
    # The exception returned by `ffprobe` is in bytes
    logging.warn("Exception probing video: {}".format(e))
  else:
    video_stream = next(
      (
        stream
        for stream in probe["streams"]
        if stream["codec_type"] == "video"
      ),
      None,
    )
    try:
      fps = float(Fraction(video_stream["avg_frame_rate"]))
    except Exception as e:
      fps = 0
    try:
      total_frames = int(video_stream["nb_frames"])
    except Exception as e:
      duration = float(video_stream['duration'])
      total_frames = int(duration*fps)
    width = video_stream["width"]
    height = video_stream["height"]
    codec = video_stream["codec_name"]
    bitrate = float(video_stream["bit_rate"])/1000.0
    return fps, total_frames, width, height, codec, bitrate

def _probe_video_frames(path):
  # ffprobe -select_streams v -show_frames video.mp4
  args = ['ffprobe', '-show_frames', '-of', 'json', '-loglevel', 'repeat+debug']
  args += [path]
  p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  communicate_kwargs = {}
  out, err = p.communicate(**communicate_kwargs)
  if p.returncode != 0:
    raise RuntimeError('ffprobe', out, err)
  return json.loads(out.decode('utf-8')), err

def identify_eoi_missing_idxs(path, total_frames):
  try:
    _, err = _probe_video_frames(path)
  except Exception as e:
    # The exception returned by `ffprobe` is in bytes
    logging.warn("Exception probing video frames: {}".format(e.stderr.decode()))
  else:
    info = [f for f in err.split(b'\n') if (FFPROBE_OK in f) or (FFPROBE_CORR in f) or (FFPROBE_INFO) in f]
    info = [0 if (FFPROBE_OK in f[0] and FFPROBE_OK in f[1]) else \
            1 if (FFPROBE_OK in f[0] and FFPROBE_CORR in f[1]) else -1 \
            for f in zip(info, info[1:] + [FFPROBE_OK])]
    info = [f for f in info if f != -1]
    assert total_frames == len(info), "len(info) {} should equal total_frames {}".format(
      len(info), total_frames)
    return info