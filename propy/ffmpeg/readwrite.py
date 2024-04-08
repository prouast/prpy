###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, September 2021              #
###############################################################################

import ffmpeg
import logging
import numpy as np
import os
from PIL import Image
import shutil
import subprocess

from propy.ffmpeg.probe import probe_video
from propy.ffmpeg.utils import find_factors_near

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

def _ffmpeg_input_from_numpy(w, h, fps, pix_fmt):
  stream = ffmpeg.input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s='{}x{}'.format(w, h), r=fps)
  return stream

def _ffmpeg_filtering(stream, fps, n, w, h, target_fps=None, crop=None, scale=None, trim=None, preserve_aspect_ratio=False, scale_algorithm='bicubic'):
  """Take an ffmpeg stream and optionally add filtering operations:
    downsampling, spatial cropping, spatial scaling (applied to result of
    cropping if specified) and temporal trimming.
  Args:
    stream: The ffmpeg stream
    fps: The original frame rate
    n: The original number of frames
    w: The original width
    h: The original height
    target_fps: Try to downsample frames to achieve this framerate
    crop: Tuple with coords and sizes for spatial cropping (x, y, width, height)
      Ignore if None.
    scale: Tuple with sizes for spatial scaling (width, height). Ignore if None.
    trim: Tuple with frame numbers for temporal trimming (start, end).
      Ignore if None.
    preserve_aspect_ratio: Preserve the aspect ratio if scaling.
    scale_algorithm: The algorithm used for scaling.
      Supported: bicubic, bilinear, area, lanczos. Default: bicubic
  Returns:
    stream: The modified ffmpeg stream
    target_n: The target number of frames
    target_w: The target width
    target_h: The target shape
    ds_factor: The applied downsampling factor
  """
  ds_factor = 1
  if target_fps > fps: logging.warn("target_fps should not be greater than fps. Ignoring.")
  elif target_fps is not None: ds_factor = int(fps // target_fps)
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
  if ds_factor > 1:
    stream = ffmpeg.filter(stream, 'select', 'not(mod(n,{}))'.format(ds_factor))
  # Cropping
  if crop is not None:
    stream = stream.crop(crop[0], crop[1], crop[2], crop[3])
  # Scaling
  # http://trac.ffmpeg.org/wiki/Scaling#Specifyingscalingalgorithm
  if scale not in [None, (0, 0)]:
    stream = ffmpeg.filter(stream, 'scale', target_w, target_h, scale_algorithm)
  # Return
  return stream, target_n, target_w, target_h, ds_factor

def _ffmpeg_output_to_numpy(stream, r, target_fps, target_n, target_w, target_h, scale=None, crf=None, pix_fmt='bgr24', preserve_aspect_ratio=False, scale_algorithm='bicubic', dim_deltas=(0, 0, 0)):
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
      preserve_aspect_ratio=preserve_aspect_ratio, scale_algorithm=scale_algorithm)
    stream = stream.output("pipe:", vsync=0, format="rawvideo", pix_fmt=pix_fmt)
    out, _ = stream.run(input=out, capture_stdout=True, capture_stderr=True)
  # Swap h and w if necessary -> not needed if scaled!
  if r != 0:
    if abs(r) == 90:
      logging.warn("Rotation {} present in video fixed; results in W and H swapped.".format(r))
      tmp = target_w
      target_w = target_h
      target_h = tmp
    else:
      logging.warn("Rotation {} present in video; Fixing is not yet supported.".format(r))
  # Parse result
  frames = np.frombuffer(out, np.uint8)
  adj_n, adh_h, adh_w = find_factors_near(
    frames.shape[0]/3, target_n, target_h, target_w, dim_deltas[0], dim_deltas[1], dim_deltas[2])
  assert adj_n * adh_h * adh_w * 3 == frames.shape[0]
  frames = frames.reshape([adj_n, adh_h, adh_w, 3])
  # Return
  return frames

def _ffmpeg_output_to_file(stream, output_dir, output_file, from_stdin=None, pix_fmt='yuv420p', crf=12, overwrite=False):
  """Run the stream and encode to file as H264"""
  output_path = os.path.join(output_dir, output_file)
  stream = ffmpeg.output(stream, output_path, pix_fmt=pix_fmt, crf=crf)
  if overwrite:
    stream = stream.global_args("-vsync", "2", "-y")
  else:
    stream = stream.global_args("-vsync", "2")
  if from_stdin is None:
    stream.run(quiet=True)
  else:
    process = stream.run_async(pipe_stdin=True, quiet=True)
    process.communicate(input=from_stdin)

def _ffmpeg_output_to_jpegs(stream, output_dir, output_file_start, target_fps, target_n, target_w, target_h, scale=None, crf=12, pix_fmt='bgr24', preserve_aspect_ratio=False):
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
      stream, fps=target_fps, n=target_n, w=target_w, h=target_h, scale=scale)
    stream = stream.output("pipe:", vsync=0, format="rawvideo", pix_fmt=pix_fmt)
    out, _ = stream.run(input=out, capture_stdout=True, capture_stderr=True)
  # Parse result
  frames = np.frombuffer(out, np.uint8)
  adj_n, adh_h, adh_w = find_factors_near(
    frames.shape[0]/3, target_n, target_h, target_w, 50, 1, 1)
  assert adj_n * adh_h * adh_w * 3 == frames.shape[0]
  frames = frames.reshape([adj_n, adh_h, adh_w, 3])
  # Write the frames
  for i, frame in enumerate(frames):
    output_file = output_file_start + "_frame_{}.jpeg".format(i)
    frame_path = os.path.join(output_dir, output_file)
    # TODO specify jpeg quality?
    Image.fromarray(frame).save(frame_path)

def read_video_from_path(path, target_fps=None, crop=None, scale=None, trim=None, crf=None, pix_fmt='bgr24', preserve_aspect_ratio=False, scale_algorithm='bicubic', order='scale_crf', dim_deltas=(0, 0, 0)):
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
    pix_fmt: Pixel format to read into
    preserve_aspect_ratio: Preserve the aspect ratio if scaling.
    scale_algorithm: The algorithm used for scaling.
      Supported: bicubic, bilinear, area, lanczos. Default: bicubic
    order: scale_crf or crf_scale - specifies order of application
    dim_deltas: Allowed deviation from target (n_franes, height, width)
  Returns:
    frames: The frames (N, H, W, 3)
    ds_factor: The applied downsampling factor
  """
  # Check if file exists
  if not os.path.exists(path):
    raise FileNotFoundError("File {} does not exist".format(path))
  # Get metadata of original video
  fps, n, w, h, _, _, r = probe_video(path=path)
  # Input
  stream = _ffmpeg_input_from_path(path=path, fps=fps, trim=trim)
  # Filtering
  scale_0 = scale if order == 'scale_crf' or crf == None else (0, 0)
  stream, target_n, target_w, target_h, ds_factor = _ffmpeg_filtering(
    stream=stream, fps=fps, n=n, w=w, h=h, target_fps=target_fps, crop=crop, scale=scale_0,
    trim=trim, preserve_aspect_ratio=preserve_aspect_ratio, scale_algorithm=scale_algorithm)
  # Save whether rotation still present
  if scale not in [None, (0, 0)] or crop is not None:
    r = 0
  # Output
  scale_1 = (0, 0) if order == 'scale_crf' or crf == None else scale
  frames = _ffmpeg_output_to_numpy(
    stream=stream, r=r, target_fps=target_fps, target_n=target_n, target_w=target_w,
    target_h=target_h, scale=scale_1, crf=crf, pix_fmt=pix_fmt, scale_algorithm=scale_algorithm,
    dim_deltas=dim_deltas)
  # Return
  return frames, ds_factor

def write_video_from_path(path, output_dir, output_file, target_fps=None, crop=None, scale=None, trim=None, pix_fmt='yuv420p', crf=12, preserve_aspect_ratio=False, scale_algorithm='bicubic', overwrite=False):
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
    scale_algorithm: The algorithm used for scaling. Default: bicubic
  """
  # Get metadata of original video
  fps, n, w, h, _, _, r = probe_video(path=path)
  # Input
  stream = _ffmpeg_input_from_path(path=path, fps=fps, trim=trim)
  # Filtering
  stream, _, _, _, _ = _ffmpeg_filtering(
      stream=stream, fps=fps, n=n, w=w, h=h, target_fps=target_fps, crop=crop, scale=scale,
      trim=trim, preserve_aspect_ratio=preserve_aspect_ratio, scale_algorithm=scale_algorithm)
  # Output
  _ffmpeg_output_to_file(
    stream, output_dir=output_dir, output_file=output_file, pix_fmt=pix_fmt, crf=crf, overwrite=overwrite)

def write_video_from_numpy(data, fps, pix_fmt, output_dir, output_file, out_pix_fmt='yuv420p', crf=12, overwrite=False):
  """Write data from a numpy array to a video file.
  Args:
    data: The numpy array with video frames. Shape [N, H, W, 3]
    fps: The frame rate
    pix_fmt: The pixel format of `data`
    output_dir: The directory where the video will be written
    ourput_file: The filename as which the video will be written
    pix_fmt: The pixel format for the output video
    crf: Constant rate factor for H.264 encoding (higher = more compression)
    overwrite: Overwrite if file exists?
  """
  _, h, w, _ = data.shape
  stream = _ffmpeg_input_from_numpy(w=w, h=h, fps=fps, pix_fmt=pix_fmt)
  buffer = data.flatten().tobytes()
  _ffmpeg_output_to_file(
    stream, output_dir=output_dir, output_file=output_file, from_stdin=buffer,
    crf=crf, pix_fmt=out_pix_fmt, overwrite=overwrite)

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
  fps, n, w, h, _, _, _ = probe_video(path=path)
  # Input
  stream = _ffmpeg_input_from_path(path=path, fps=fps, trim=trim)
  # Filtering
  scale_0 = scale if order == 'scale_crf' or crf == None else (0, 0)
  stream, target_n, target_w, target_h, _ = _ffmpeg_filtering(
    stream=stream, fps=fps, n=n, w=w, h=h, target_fps=target_fps, crop=crop, scale=scale_0,
    trim=trim, preserve_aspect_ratio=preserve_aspect_ratio)
  # Output
  scale_1 = (0, 0) if order == 'scale_crf' or crf == None else scale
  _ffmpeg_output_to_jpegs(
    stream=stream, output_dir=output_dir, output_file_start=output_file_start, target_fps=target_fps,
    target_n=target_n, target_w=target_w, target_h=target_h, scale=scale_1, crf=crf)

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
  fps, _, _, _, _, _, _ = probe_video(path)
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
