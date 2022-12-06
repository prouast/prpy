###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, September 2021              #
###############################################################################

import ffmpeg
from fractions import Fraction
import itertools
import json
import logging
import numpy as np
import os
from PIL import Image
import shutil
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
  return out, err

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

def _find_factors_near(i, f1, f2, f3, s1, s2, s3):
  ts = [(t1, t2, t3) for t1, t2, t3 in list(itertools.product( \
    range(f1-s1, f1+s1+1), range(f2-s2, f2+s2+1), range(f3-s3, f3+s3+1)))]
  for t1, t2, t3 in ts:
    if t1 * t2 * t3 == i:
      return t1, t2, t3
  raise RuntimeError("Could not find factors near the provided values")

def _ffmpeg_input_from_path(path, fps, trim):
  """Use file as input part of ffmpeg command"""
  trim_start = 0 if trim is None else trim[0]
  # Create the stream
  stream = ffmpeg.input(filename=path, ss=trim_start/fps)
  # Return
  return stream

def _ffmpeg_input_from_pipe():
  """Use pipe as input part of ffmpeg command"""
  stream = ffmpeg.input("pipe:")
  return stream

def _ffmpeg_filtering(stream, fps, n, w, h, target_fps=None, crop=None, scale=None, trim=None, preserve_aspect_ratio=False):
  """Take an ffmpeg stream and optionally add filtering operations:
    downsampling, spatial cropping, spatial scaling (applied to result of
    cropping if specified) and temporal trimming.
  Args:
    stream: The ffmpeg stream
    target_fps: Try to downsample frames to achieve this framerate
    crop: Tuple with coords and sizes for spatial cropping (x, y, width, height)
      Ignore if None.
    scale: Tuple with sizes for spatial scaling (width, height). Ignore if None.
    trim: Tuple with frame numbers for temporal trimming (start, end).
      Ignore if None.
    preserve_aspect_ratio: Preserve the aspect ratio if scaling.
  Returns:
    stream: The modified ffmpeg stream
    target_n: The target number of frames
    target_w: The target width
    target_h: The target shape
    ds_factor: The applied downsampling factor
  """
  assert target_fps is None or target_fps <= fps, "target_fps cannot be greater than fps"
  # Process downsampling settings
  ds_factor = 1 if target_fps is None else int(fps // target_fps)
  # Target number of frames
  target_n = trim[1] - trim[0] if trim is not None else n
  target_n = int(target_n / ds_factor)
  # Target size after taking into account cropping
  target_w = crop[2] if crop is not None else w
  target_h = crop[3] if crop is not None else h
  # Target size after taking into account scaling
  if scale not in [None, (0, 0)]:
    if preserve_aspect_ratio:
      scale_ratio = max(scale) / max(target_w, target_h)
      target_w = int(target_w * scale_ratio)
      target_h = int(target_h * scale_ratio)
    else:
      target_w, target_h = scale
  # Trimming
  if trim is not None:
    stream = stream.trim(start_frame=0, end_frame=trim[1]-trim[0])
    stream = stream.setpts('PTS-STARTPTS')
  # Downsampling
  if target_fps is not None:
    stream = ffmpeg.filter(stream, 'select', 'not(mod(n,{}))'.format(ds_factor))
  # Cropping
  if crop is not None:
    stream = stream.crop(crop[0], crop[1], crop[2], crop[3])
  # Scaling
  if scale not in [None, (0, 0)]:
    stream = ffmpeg.filter(stream, 'scale', target_w, target_h, 'bicubic')
  # Return
  return stream, target_n, target_w, target_h, ds_factor

def _ffmpeg_output_to_numpy(stream, target_fps, target_n, target_w, target_h, scale=None, crf=None, pix_fmt='bgr24', preserve_aspect_ratio=False):
  """Run the stream and capture the raw video output in a numpy array"""
  if crf is None:
    # Run stream straight to raw video
    stream = stream.output("pipe:", vsync=0, format="rawvideo", pix_fmt=pix_fmt)
    stream = stream.global_args("-loglevel", "panic", "-hide_banner", "-nostdin", "-nostats")
    out, _ = stream.run(capture_stdout=True, capture_stderr=True)
  else:
    # Run stream to encode H264 with crf
    stream = stream.output("pipe:", vsync=0, format='rawvideo', vcodec='libx264', crf=crf)
    out, _ = stream.run(capture_stdout=True, capture_stderr=True)
    # Run stream to decode H264 to raw video
    stream = _ffmpeg_input_from_pipe()
    stream, _, target_w, target_h, _ = _ffmpeg_filtering(
      stream, fps=target_fps, n=target_n, w=target_w, h=target_h, scale=scale,
      preserve_aspect_ratio=preserve_aspect_ratio)
    stream = stream.output("pipe:", vsync=0, format="rawvideo", pix_fmt=pix_fmt)
    out, _ = stream.run(input=out, capture_stdout=True, capture_stderr=True)
  # Parse result
  frames = np.frombuffer(out, np.uint8)
  adj_n, adh_h, adh_w = _find_factors_near(
    frames.shape[0]/3, target_n, target_h, target_w, 2, 1, 1)
  assert adj_n * adh_h * adh_w * 3 == frames.shape[0]
  frames = frames.reshape([adj_n, adh_h, adh_w, 3])
  # Return
  return frames

def _ffmpeg_output_to_file(stream, output_dir, output_file, crf=12):
  """Run the stream and encode to file as H264"""
  output_path = os.path.join(output_dir, output_file)
  stream = ffmpeg.output(stream, output_path, crf=crf)
  stream = stream.global_args("-vsync", "2")
  stream.run(quiet=True)

def _ffmpeg_output_to_jpegs(stream, output_dir, output_file_start, target_fps, target_n, target_w, target_h, scale_w=None, scale_h=None, crf=12, pix_fmt='bgr24', preserve_aspect_ratio=False):
  if crf is None:
    # Run stream straight to raw video
    stream = stream.output("pipe:", vsync=0, format="rawvideo", pix_fmt=pix_fmt)
    stream = stream.global_args("-loglevel", "panic", "-hide_banner", "-nostdin", "-nostats")
    out, _ = stream.run(capture_stdout=True, capture_stderr=True)
  else:
    # Run stream to encode H264 with crf
    stream = stream.output("pipe:", vsync=0, format='rawvideo', vcodec='libx264', crf=crf)
    out, _ = stream.run(capture_stdout=True, capture_stderr=True)
    # Run stream to decode H264 to raw video
    stream = _ffmpeg_input_from_pipe()
    stream, _, target_w, target_h, _ = _ffmpeg_filtering(
      stream, fps=target_fps, n=target_n, w=target_w, h=target_h, scale_w=scale_w, scale_h=scale_h)
    stream = stream.output("pipe:", vsync=0, format="rawvideo", pix_fmt=pix_fmt)
    out, _ = stream.run(input=out, capture_stdout=True, capture_stderr=True)
  # Parse result
  frames = np.frombuffer(out, np.uint8)
  adj_n, adh_h, adh_w = _find_factors_near(
    frames.shape[0]/3, target_n, target_h, target_w, 2, 1, 1)
  assert adj_n * adh_h * adh_w * 3 == frames.shape[0]
  frames = frames.reshape([adj_n, adh_h, adh_w, 3])
  # Write the frames
  for i, frame in enumerate(frames):
    output_file = output_file_start + "_frame_{}.jpeg".format(i)
    frame_path = os.path.join(output_dir, output_file)
    # TODO specify jpeg quality?
    Image.fromarray(frame).save(frame_path)

def read_video_from_path(path, target_fps=None, crop=None, scale=None, trim=None, crf=None, pix_fmt='bgr24', preserve_aspect_ratio=False, order='scale_crf'):
  """Read a video from path into a numpy array, optionally transformed by
    downsampling, spatial cropping, spatial scaling (applied to result of
    cropping if specified), temporal trimming, and intermediate encoding.
  Args:
    path: The path from which video will be read.
    target_fps: Try to downsample frames to achieve this framerate
    crop: Tuple with coords and sizes for spatial cropping (x, y, width, height)
      Ignore if None.
    scale: Tuple with sizes for spatial scaling (width, height). Ignore if None.
    trim: Tuple with frame numbers for temporal trimming (start, end).
      Ignore if None.
    crf: Constant rate factor for H.264 encoding (higher = more compression)
      If specified, do intermediate encoding, otherwise ignore.
    pix_fmt: Pixel format
    preserve_aspect_ratio: Preserve the aspect ratio if scaling.
    order: scale_crf or crf_scale - specifies order of application
  Returns:
    frames: The frames [N, H, W, 3]
    ds_factor: The applied downsampling factor
  """
  # Get metadata of original video
  fps, n, w, h, _, _ = probe_video(path)
  # Input
  stream = _ffmpeg_input_from_path(path, fps, trim)
  # Filtering
  scale_0 = scale if order == 'scale_crf' or crf == None else (0, 0)
  stream, target_n, target_w, target_h, ds_factor = _ffmpeg_filtering(
    stream, fps, n, w, h, target_fps, crop, scale_0, trim, preserve_aspect_ratio)
  # Output
  scale_1 = (0, 0) if order == 'scale_crf' or crf == None else scale
  frames = _ffmpeg_output_to_numpy(stream, target_fps, target_n, target_w, target_h, scale_1, crf, pix_fmt)
  # Return
  return frames, ds_factor

def write_video_from_path(path, output_dir, output_file, target_fps=None, crop=None, scale=None, trim=None, crf=12, preserve_aspect_ratio=False):
  """Read a video from path and write back to a video file, optionally
    transformed by downsampling, spatial cropping, spatial scaling (applied to
    result of cropping if specified), and temporal trimming.
  Args:
    path: The path from which video will be read.
    output_dir: The directory where the video will be written
    ourput_file: The filename as which the video will be written
    target_fps: Try to downsample frames to achieve this framerate
    crop: Tuple with coords and sizes for spatial cropping (x, y, width, height)
      Ignore if None.
    scale: Tuple with sizes for spatial scaling (width, height). Ignore if None.
    trim: Tuple with frame numbers for temporal trimming (start, end).
      Ignore if None.
    crf: Constant rate factor for H.264 encoding (higher = more compression)
    preserve_aspect_ratio: Preserve the aspect ratio if scaling.
  """
  # Get metadata of original video
  fps, n, w, h, _, _ = probe_video(path)
  # Input
  stream = _ffmpeg_input_from_path(path, fps, trim)
  # Filtering
  stream, _, _, _, _ = _ffmpeg_filtering(
      stream, fps, n, w, h, target_fps, crop, scale, trim, preserve_aspect_ratio)
  # Output
  _ffmpeg_output_to_file(stream, output_dir, output_file, crf)

def write_jpegs_from_path(path, output_dir, output_file_start, target_fps=None, crop=None, scale=None, trim=None, crf=None, preserve_aspect_ratio=False, order='scale_crf'):
  """Read a video from path and write back as jpegs, optionally transformed by
    downsampling, spatial cropping, spatial scaling (applied to result of
    cropping if specified), temporal trimming, and intermediate encoding.
  Args:
    path: The path from which video will be read.
    output_dir: The directory where the jpegs will be written
    output_file_start: The filename as which the jpegs will be written
    target_fps: Try to downsample frames to achieve this framerate
    crop: Tuple with coords and sizes for spatial cropping (x, y, width, height)
      Ignore if None.
    scale: Tuple with sizes for spatial scaling (width, height). Ignore if None.
    trim: Tuple with frame numbers for temporal trimming (start, end).
      Ignore if None.
    crf: Constant rate factor for H.264 encoding (higher = more compression)
      If specified, do intermediate encoding, otherwise ignore.
    preserve_aspect_ratio: Preserve the aspect ratio if scaling.
    order: scale_crf or crf_scale - specifies order of application
  """
  # Get metadata of original video
  fps, n, w, h, _, _ = probe_video(path)
  # Input
  stream = _ffmpeg_input_from_path(path, fps, trim)
  # Filtering
  scale_0 = scale if order == 'scale_crf' or crf == None else (0, 0)
  stream, target_n, target_w, target_h, ds_factor = _ffmpeg_filtering(
    stream, fps, n, w, h, target_fps, crop, scale_0, trim, preserve_aspect_ratio)
  # Output
  scale_1 = (0, 0) if order == 'scale_crf' or crf == None else scale
  _ffmpeg_output_to_jpegs(
    stream, output_dir, output_file_start, target_fps, target_n, target_w,
    target_h, scale_1, crf)

def stream_to_mp4_container(stream_filename, video_filename, framerate, delete_stream=False):
  """Like ffmpeg -framerate 60 -i input.h264 -c copy output.mp4"""
  stream = ffmpeg.input(stream_filename, framerate=framerate)
  stream = ffmpeg.output(stream, video_filename, vcodec='copy')
  stream = stream.global_args("-loglevel", "panic", "-hide_banner")
  ffmpeg.run(stream)
  if delete_stream:
    os.remove(stream_filename)

def write_lossless_trim_and_crop_video(path, output_path, start, end, x, y, width, height):
  """Lossless trim and crop for MJPEG video.
    Quite slow because could not get jpegtran bindings running on macOS.
  """
  fps, _, _, _, _, _ = probe_video(path)
  # https://stackoverflow.com/questions/33378548/ffmpeg-crop-a-video-without-losing-the-quality
  logging.debug("Lossless crop {} between start={} and end={}".format(
    path, start, end))
  # 1a. Trim the video in time and export all individual jpeg with ffmpeg + mjpeg2jpeg
  # https://superuser.com/questions/377343/cut-part-from-video-file-from-start-position-to-end-position-with-ffmpeg
  # ffmpeg -ss [start] -i in.mp4 -t [duration] -c copy out.mp4
  # ffmpeg -i mjpeg-movie.avi -c:v copy -bsf:v mjpeg2jpeg frames_%d.jpg
  jpeg_folder = os.path.splitext(output_path)[0]
  if not os.path.exists(jpeg_folder):
    os.makedirs(jpeg_folder)
  jpeg_path = os.path.join(jpeg_folder, "frame_%03d.jpg")
  stream = ffmpeg.input(path, ss=start/fps, t=(end-start)/fps)
  stream = ffmpeg.output(stream, jpeg_path, vcodec='copy', **{'bsf:v': 'mjpeg2jpeg'})
  stream.run(quiet=True)
  # 1b. Count to make sure the number of frames equals chunk.frame_number
  jpegs = os.listdir(jpeg_folder)
  jpeg_count = len(jpegs)
  if jpeg_count != end - start:
    logging.debug("Number of jpegs ({}) != chunk.frame_count ({})".format(
      jpeg_count, end-start))
    drop_count = jpeg_count - (end-start)
    logging.debug("Dropping {} jpegs...".format(drop_count))
    for filename in sorted(jpegs)[-drop_count:]:
      filepath = os.path.join(jpeg_folder, filename)
      os.remove(filepath)
  # 2 Crop all individual jpeg with jpegtran
  # jpegtran -perfect -crop WxH+X+Y -outfile crop.jpg image.jpg
  # https://ffmpeg.org/ffmpeg-bitstream-filters.html#mjpeg2jpeg
  # TODO: Use jpegtran bindings/library on linux for speedup
  for filename in os.listdir(jpeg_folder):
    filepath = os.path.join(jpeg_folder, filename)
    out_filepath = os.path.splitext(filepath)[0] + "_c.jpg"
    subprocess.call(
      "jpegtran -perfect -crop {}x{}+{}+{} -outfile {} {}".format(
        width, height, x, y, out_filepath, filepath), shell=True)
    os.remove(filepath)
  # 3. Join individual jpg back together
  # ffmpeg -framerate 59.94001960164264 -i frame_%d.jpg -c:v copy back.mp4
  cropped_jpeg_path = os.path.join(jpeg_folder, "frame_%03d_c.jpg")
  stream = ffmpeg.input(cropped_jpeg_path, framerate=fps)
  stream = ffmpeg.output(stream, output_path, vcodec='copy')
  stream.run(quiet=True)
  # 4. Delete jpeg directory
  shutil.rmtree(jpeg_folder)
