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
import logging
import math
import numpy as np
import os
import re
from typing import Tuple, Union

from prpy.ffmpeg.probe import probe_video
from prpy.ffmpeg.utils import find_factors_near

def _ffmpeg_input_from_path(
    path: str,
    fps: Union[float, int],
    trim: tuple
  ) -> ffmpeg.nodes.FilterableStream:
  """Use file as input part of ffmpeg command.
  
  Args:
    path: The path from which video will be read.
    fps: The framerate of the input video.
    trim: Frame numbers for temporal trimming (start, end).
  Returns:
    stream: ffmpeg input stream from file
  """
  assert isinstance(path, str)
  assert isinstance(fps, (float, int))
  assert trim is None or (isinstance(trim, tuple) and all(isinstance(i, int) for i in trim))
  trim_start = 0 if trim is None else trim[0]
  # Create the stream and skip forward to trim start if applicable
  stream = ffmpeg.input(filename=path, ss=trim_start/fps)
  # Return
  return stream

def _ffmpeg_input_from_pipe() -> ffmpeg.nodes.FilterableStream:
  """Use pipe as input part of ffmpeg command.
  
  Returns:
    stream: ffmpeg input stream from pipe
  """
  stream = ffmpeg.input("pipe:")
  return stream

def _ffmpeg_input_from_numpy(
    w: int,
    h: int,
    fps: Union[float, int],
    pix_fmt: str
  ) -> ffmpeg.nodes.FilterableStream:
  """Use file as input part of ffmpeg command.
  
  Args:
    w: The width dimension of the video data from numpy.
    h: The height dimension of the video data from numpy.
    fps: The framerate of the video data from numpy.
    pix_fmt: The pixel format of the video data from numpy (e.g., `bgr24`).
  Returns:
    stream: ffmpeg input stream from pipe (numpy)
  """
  assert isinstance(w, int)
  assert isinstance(h, int)
  assert isinstance(fps, (float, int))
  assert isinstance(pix_fmt, str)
  stream = ffmpeg.input('pipe:', format='rawvideo', pix_fmt=pix_fmt, s=f'{w}x{h}', r=fps)
  return stream

def _ffmpeg_filtering(
    stream: ffmpeg.nodes.FilterableStream,
    fps: Union[float, int],
    n: int,
    w: int,
    h: int,
    target_fps: Union[float, int, None] = None,
    crop: Union[tuple, None] = None,
    scale: Union[Tuple[int, tuple], None] = None,
    trim: Union[tuple, None] = None,
    preserve_aspect_ratio: bool = False,
    scale_algorithm: str = 'bicubic',
    requires_even_dims: bool = False
  ) -> Tuple[ffmpeg.nodes.FilterableStream, int, int, int, int]:
  """Take an ffmpeg stream and optionally add filtering operations.
  Downsampling, spatial cropping, spatial scaling (applied to result of
    cropping if specified) and temporal trimming.

  Args:
    stream: The ffmpeg stream
    fps: The existing frame rate
    n: The existing number of frames
    w: The existing width
    h: The existing height
    target_fps: Downsample frames to approximate this framerate (optional)
    crop: Coords for spatial cropping (x0, y0, x1, y1) (optional)
    scale: Size(s) for spatial scaling. Scalar or (width, height) (optional)
    trim: Frame numbers for temporal trimming (start, end) (optional)
    preserve_aspect_ratio: Preserve the aspect ratio if scaling
    scale_algorithm: The algorithm used for scaling.
      Supported: bicubic, bilinear, area, lanczos. Default: bicubic
    requires_even_dims: target_w and target_h must be divisible by 2
  Returns:
    Tuple of
     - stream: The modified ffmpeg stream
     - target_n: The target number of frames
     - target_w: The target width
     - target_h: The target shape
     - ds_factor: The applied downsampling factor
  """
  assert isinstance(stream, ffmpeg.nodes.FilterableStream)
  assert isinstance(fps, (float, int))
  assert isinstance(n, int)
  assert isinstance(w, int)
  assert isinstance(h, int)
  assert target_fps is None or isinstance(target_fps, (float, int))
  assert crop is None or (isinstance(crop, tuple) and len(crop) == 4 and all(isinstance(i, (int, np.int64, np.int32)) for i in crop))
  assert scale is None or isinstance(scale, int) or (isinstance(scale, tuple) and len(scale) == 2 and all(isinstance(i, int) for i in scale))
  assert trim is None or (isinstance(trim, tuple) and len(trim) == 2 and all(isinstance(i, int) for i in trim))
  assert isinstance(preserve_aspect_ratio, bool)
  assert isinstance(scale_algorithm, str)
  ds_factor = 1
  if target_fps is not None and target_fps > fps: logging.debug("target_fps should not be greater than fps. Ignoring.")
  elif target_fps is not None: ds_factor = round(fps / target_fps)
  # Target number of frames
  target_n = trim[1] - trim[0] if trim is not None else n
  target_n = math.ceil(target_n / ds_factor)
  # If required, pad the crop to make target dimensions even
  if requires_even_dims and crop is not None and scale in [None, 0]:
    if (crop[2] - crop[0]) % 2 != 0:
      crop = (crop[0], crop[1], crop[2]-1, crop[3])
      logging.warning(f"Reducing uneven crop width from {crop[2]+1-crop[0]} to {crop[2]-crop[0]} to make operation possible.")
    if (crop[3] - crop[1]) % 2 != 0:
      crop = (crop[0], crop[1], crop[2], crop[3]-1)
      logging.warning(f"Reducing uneven crop height from {crop[3]+1-crop[1]} to {crop[3]-crop[1]} to make operation possible.")
  # Target size after taking into account cropping
  target_w = crop[2]-crop[0] if crop is not None else w
  target_h = crop[3]-crop[1] if crop is not None else h
  # Target size after taking into account scaling
  if scale not in [None, 0]:
    if isinstance(scale, int): scale = (scale, scale)
    if preserve_aspect_ratio:
      scale_ratio = max(scale) / max(target_w, target_h)
      target_w = round(target_w * scale_ratio)
      target_h = round(target_h * scale_ratio)
    else:
      target_w, target_h = scale
    if requires_even_dims and (target_h % 2 != 0 or target_w % 2 != 0):
      raise ValueError(f"Cannot use this scale ({scale}) because encoding requires even height and width.")
  # Trimming
  if trim is not None:
    # Node: This works because we skipped forward to trim[0] when creating the stream.
    stream = stream.trim(start_frame=0, end_frame=trim[1]-trim[0])
    stream = stream.setpts('PTS-STARTPTS')
  # Downsampling
  if ds_factor > 1:
    stream = ffmpeg.filter(stream, 'select', f'not(mod(n,{ds_factor}))')
    stream = stream.setpts('N/FRAME_RATE/TB')
  # Cropping
  if crop is not None:
    stream = stream.crop(crop[0], crop[1], crop[2]-crop[0], crop[3]-crop[1])
  # Scaling
  # http://trac.ffmpeg.org/wiki/Scaling#Specifyingscalingalgorithm
  if scale not in [None, (0, 0)]:
    stream = ffmpeg.filter(stream, 'scale', target_w, target_h, scale_algorithm)
  # Return
  return stream, target_n, target_w, target_h, ds_factor

def _ffmpeg_output_to_numpy(
    stream: ffmpeg.nodes.FilterableStream,
    r: int,
    fps: Union[float, int, None],
    n: int,
    w: Union[int, np.int64, np.int32],
    h: Union[int, np.int64, np.int32],
    scale: Union[tuple, int, None] = None,
    crf: Union[int, None] = None,
    pix_fmt: str = 'bgr24',
    preserve_aspect_ratio: bool = False,
    scale_algorithm: str = 'bicubic',
    dim_deltas: tuple = (0, 0, 0),
    quiet: bool = True
  ) -> np.ndarray:
  """Run the stream and capture the raw video output in a numpy array.
  
  Args:
    stream: The ffmpeg stream
    r: Rotation present (in terms of flipped w and h)
    fps: Framerate attempted to create in stream.
    n: Number of frames attempted to create in stream.
    w: Frame width attempted to create in stream.
    h: Frame height attempted to create in stream.
    scale: Size(s) for spatial scaling. Scalar or (width, height) (optional)
    crf: Constant rate factor for H.264 encoding (higher = more compression)
      If not None, need to run the stream to encode before capturing to np.
    pix_fmt: Pixel format to read into.
    preserve_aspect_ratio: Preserve the aspect ratio if scaling
    scale_algorithm: The algorithm used for scaling.
      Supported: bicubic, bilinear, area, lanczos. Default: bicubic
    dim_deltas: Allowed deviation from target (n_frames, height, width)
    quiet: Whether to suppress ffmpeg log
  Returns:
    frames: The video frames in shape (n, h, w, c)
  """
  assert isinstance(stream, ffmpeg.nodes.FilterableStream)
  assert isinstance(r, int)
  assert fps is None or isinstance(fps, (float, int))
  assert isinstance(n, int)
  assert isinstance(w, (int, np.int64, np.int32))
  assert isinstance(h, (int, np.int64, np.int32))
  assert crf is None or isinstance(crf, int)
  assert isinstance(pix_fmt, str)
  assert isinstance(dim_deltas, tuple) and len(dim_deltas) == 3 and all(isinstance(i, int) for i in dim_deltas)
  if crf is None:
    # Stream straight to raw video. Absorps any rotation metadata present in the input video into pixels.
    stream = stream.output("pipe:", vsync="passthrough", format="rawvideo", pix_fmt=pix_fmt)
    out, err = stream.run(capture_stdout=True, capture_stderr=True)
    _, _, out_r = _read_output_stream_properties(err, quiet=quiet)
    assert out_r == 0, "Rotation should be 0 at this point"
  else:
    # Run stream to encode H264 with crf.
    # Explicitly specify output dimensions so rotation metadata present in the input video is correctly absorbed into pixels.
    w, h, r = _rectify_w_h_rotation(w, h, r)
    stream = stream.output("pipe:", vsync="passthrough", format='rawvideo', vcodec='libx264', s=f"{w}x{h}", crf=crf)
    out, err = stream.run(capture_stdout=True, capture_stderr=True)
    _, _, out_r = _read_output_stream_properties(err, quiet=quiet)
    # Run stream to decode H264 to raw video
    stream = _ffmpeg_input_from_pipe()
    stream, _, w, h, _ = _ffmpeg_filtering(
      stream, fps=fps, n=n, w=w, h=h, scale=scale,
      preserve_aspect_ratio=preserve_aspect_ratio, scale_algorithm=scale_algorithm)
    stream = stream.output("pipe:", vsync='passthrough', format="rawvideo", pix_fmt=pix_fmt)
    out, _ = stream.run(input=out, capture_stdout=True, capture_stderr=True, quiet=quiet)
    _, _, out_r = _read_output_stream_properties(err, quiet=quiet)
    assert out_r == 0, "Rotation should be 0 at this point"
  # Rectify h and w if necessary
  w, h, r = _rectify_w_h_rotation(w, h, r)
  # Parse result
  frames = np.frombuffer(out, np.uint8)
  try:
    adj_n, adh_h, adh_w = find_factors_near(
      frames.shape[0]//3, int(n), int(h), int(w), dim_deltas[0], dim_deltas[1], dim_deltas[2])
  except:
    raise ValueError("ffmpeg was not able to read the video into the expected shape using the requested settings. There may be an issue with the video file.")
  assert adj_n * adh_h * adh_w * 3 == frames.shape[0]
  frames = frames.reshape([adj_n, adh_h, adh_w, 3])
  # Return
  return frames

def _ffmpeg_output_to_file(
    stream: ffmpeg.nodes.FilterableStream,
    output_dir: str,
    output_file: str,
    from_stdin: Union[bytes, None] = None,
    pix_fmt: str = 'yuv420p',
    codec: str = 'h264',
    crf: int = 12,
    overwrite: bool = False,
    quiet: bool = True,
    preset: str = 'medium',
    hwaccel: Union[None, str] = None
  ):
  """Run the stream and encode to file with selected codec.
  
  Args:
    stream: The ffmpeg stream
    output_dir: Directory for output video
    output_file: Filename for the output video
    from_stdin: Optional bytes input for ffmpeg
    pix_fmt: Pixel format ('yuv420p', 'yuv444p', etc.)
    codec: Video codec ('h264', 'h265', 'mjpeg', 'ffv1')
    crf: Constant rate factor for h264 encoding (higher = more compression)
    overwrite: Overwrite if file exists?
    quiet: Suppress ffmpeg output?
    preset: Encoding speed preset (faster encoding vs. compression efficiency)
    hwaccel: Pass 'nvidia' to enable hardware acceleration for h264 or h265
  """
  assert isinstance(stream, ffmpeg.nodes.FilterableStream)
  assert isinstance(output_dir, str)
  assert isinstance(output_file, str)
  assert from_stdin is None or isinstance(from_stdin, bytes)
  assert isinstance(pix_fmt, str)
  assert codec in ['h264', 'h265', 'mjpeg', 'ffv1'], \
         "Codec must be 'h264', 'h265', 'mjpeg', or 'ffv1'"
  assert codec in ['h264', 'h265'] or preset == 'medium', \
         "Preset is relevant only for h264/h265"
  assert isinstance(crf, int), "Must specify crf as integer"
  assert isinstance(overwrite, bool)
  assert hwaccel in [None, 'nvidia']
  output_path = os.path.join(output_dir, output_file)
  # Determine vcodec based on codec and hardware acceleration.
  vcodecs = {
    'h264': 'h264_nvenc' if hwaccel == 'nvidia' else 'libx264',
    'h265': 'hevc_nvenc' if hwaccel == 'nvidia' else 'libx265',
    'mjpeg': 'mjpeg',
    'ffv1': 'ffv1'
  }
  vcodec = vcodecs[codec]
  # Build encoder options.
  kwargs = {}
  if codec == 'h264':
    kwargs = {
      'pix_fmt': pix_fmt,
      'crf': crf,
      'vcodec': vcodec,
      'preset': preset
    }
  elif codec == 'h265':
    if hwaccel == 'nvidia':
      kwargs = {
        'pix_fmt': pix_fmt,
        'vcodec': vcodec,
        'preset': preset,
        'rc': 'constqp',
        'qp': crf
      }
    else:
      kwargs = {
        'pix_fmt': pix_fmt,
        'crf': crf,
        'vcodec': vcodec,
        'preset': preset,
        'x265-params': 'lossless=0'
      }
  elif codec == 'mjpeg':
    kwargs = {
      'pix_fmt': 'yuvj420p',
      'vcodec': vcodec,
      "q:v": crf
    }
  elif codec == 'ffv1':
    kwargs = {
      'pix_fmt': 'yuv444p',
      'vcodec': vcodec,
    }
  # Add the output.
  stream = ffmpeg.output(stream, output_path, **kwargs)
  # Set up global args.
  global_args = ["-vsync", "passthrough"]
  if overwrite:
    global_args.append("-y")
  stream = stream.global_args(*global_args)
  # Run the stream.
  if from_stdin is None:
    stream.run(quiet=quiet, capture_stderr=quiet)
  else:
    process = stream.run_async(pipe_stdin=True, quiet=quiet)
    process.communicate(input=from_stdin)

def _rectify_w_h_rotation(
    w: int,
    h: int,
    r: int
  ) -> Tuple[int, int, int]:
  """Rectify any rotation in the order of w and h

  Args:
    w: The width with possible rotation
    h: The height with possible rotation
    r: The current rotation regarding w and h
  Returns:
    Tuple of
     - w: The rectified width
     - h: The rectified height
     - r: The updated zero rotation
  """
  if r != 0:
    if abs(r) == 90:
      logging.debug(f"Rotation {r} present in video fixed; results in W and H swapped.")
      w, h = h, w
      r = 0
    else:
      logging.warning(f"Rotation {r} present in video; Fixing is not yet supported.")
  return w, h, r

def _read_output_stream_properties(err: bytes, quiet: bool = True) -> Tuple[int, int, int]:
    """Extract output stream properties (width, height, rotation) from ffmpeg stderr log.

    Args:
      err: The stderr output from ffmpeg (as bytes) containing log information.
      quiet: If False, prints the decoded log output to the console.
    Returns:
      A tuple of:
       - w: The output width (int)
       - h: The output height (int)
       - r: The output rotation in degrees (int)
    """
    log_text = err.decode('utf-8', errors='replace')
    if not quiet:
      print(log_text)
    w, h, r = 0, 0, 0
    # Extract dimensions from a line like "Stream # ... Video: ... 1080x1920"
    dim_match = re.search(r"Output #.*?Stream #.*?Video:.*?([1-9]\d*)x([1-9]\d*)", log_text, re.DOTALL)
    if dim_match:
      w = int(dim_match.group(1))
      h = int(dim_match.group(2))
    else:
      logging.warning("Could not determine output dimensions from ffmpeg log.")
    # Extract rotation from a line like "displaymatrix: rotation of -90.00 degrees"
    rot_match = re.search(r"Output #.*?displaymatrix:\s*rotation of\s*([-\d.]+)\s*degrees", log_text)
    if rot_match:
      try:
        r = int(round(float(rot_match.group(1))))
      except ValueError:
        logging.warning("Could not parse rotation value from ffmpeg log.")
    return w, h, r

def read_video_from_path(
    path: str,
    target_fps: Union[float, None] = None,
    crop: Union[tuple, None] = None,
    scale: Union[int, tuple, None] = None,
    trim: Union[tuple, None] = None,
    crf: Union[int, None] = None,
    pix_fmt: str = 'bgr24',
    preserve_aspect_ratio: bool = False,
    scale_algorithm: str = 'bicubic',
    order: str = 'scale_crf',
    dim_deltas: tuple = (40, 0, 0),
    quiet: bool = True
  ) -> Tuple[np.ndarray, int]:
  """Read a video from path into a numpy array.
  Optionally transformed by downsampling, spatial cropping, spatial scaling
    (applied to result of cropping if specified), temporal trimming, and
    intermediate encoding.

  Args:
    path: The path from which video will be read.
    target_fps: Try to downsample frames to achieve this framerate.
    crop: Coords for spatial cropping (x0, y0, x1, y1) (optional).
    scale: Size(s) for spatial scaling. Scalar or (width, height) (optional).
    trim: Frame numbers for temporal trimming (start, end) (optional).
    crf: Constant rate factor for H.264 encoding (higher = more compression)
      If specified, do intermediate encoding, otherwise ignore.
    pix_fmt: Pixel format to read into.
    preserve_aspect_ratio: Preserve the aspect ratio if scaling.
    scale_algorithm: The algorithm used for scaling.
      Supported: bicubic, bilinear, area, lanczos. Default: bicubic
    order: scale_crf or crf_scale - specifies order of application
    dim_deltas: Allowed deviation from target (n_frames, height, width)
    quiet: Whether to suppress ffmpeg output
  Returns:
    Tuple of
     - frames: The video frames (n, h, w, 3)
     - ds_factor: The applied downsampling factor
  """
  assert isinstance(path, str)
  # Check if file exists
  if not os.path.exists(path):
    raise FileNotFoundError(f"File {path} does not exist")
  # Get metadata of original video
  fps, n, w, h, _, _, r, _ = probe_video(path=path)
  # Input
  stream = _ffmpeg_input_from_path(path=path, fps=fps, trim=trim)
  # Filtering
  scale_0 = scale if order == 'scale_crf' or crf == None else 0
  stream, target_n, target_w, target_h, ds_factor = _ffmpeg_filtering(
    stream=stream, fps=fps, n=n, w=w, h=h, target_fps=target_fps, crop=crop, scale=scale_0,
    trim=trim, preserve_aspect_ratio=preserve_aspect_ratio, scale_algorithm=scale_algorithm,
    requires_even_dims=(crf is not None) or (crop is not None and scale in [None, 0]))
  # Save whether rotation (in terms of flipped w and h) still present
  if scale_0 not in [None, 0] or crop is not None: r = 0
  # Output
  fps = target_fps if target_fps is not None else fps
  scale_1 = 0 if order == 'scale_crf' or crf == None else scale
  frames = _ffmpeg_output_to_numpy(
    stream=stream, r=r, fps=fps, n=target_n, w=target_w, h=target_h,
    scale=scale_1, crf=crf, pix_fmt=pix_fmt, scale_algorithm=scale_algorithm,
    dim_deltas=dim_deltas, quiet=quiet)
  # Check the number of frames
  expected_n = math.ceil(((trim[1]-trim[0]) if trim is not None else n) / ds_factor)
  if frames.shape[0] < expected_n:
    logging.warning(f"Less frames received than expected (delta = {frames.shape[0]-expected_n}) - this may indicate an issue with the video file. Padding to avoid issues.")
    frames = np.concatenate((np.repeat(frames[:1], expected_n - frames.shape[0], axis=0), frames), axis=0)
  elif frames.shape[0] > expected_n:
    logging.warning(f"More frames received than expected (delta = {frames.shape[0]-expected_n}) - this may indicate an issue with the video file. Trimming to avoid issues.")
    frames = frames[:expected_n]
  # Return
  return frames, ds_factor

def write_video_from_path(
    path: str,
    output_dir: str,
    output_file: str,
    target_fps: Union[float, None] = None,
    crop: Union[tuple, None] = None,
    scale: Union[int, tuple, None] = None,
    trim: Union[tuple, None] = None,
    pix_fmt: str = 'yuv420p',
    codec: str = 'h264',
    crf: int = 12,
    preserve_aspect_ratio: bool = False,
    scale_algorithm: str = 'bicubic',
    overwrite: bool = False,
    quiet: bool = True,
    preset: str = 'medium',
    hwaccel: Union[None, str] = None
  ):
  """Read a video from path and write back to a video file.
  Optionally transformed by downsampling, spatial cropping, spatial scaling
    (applied to result of cropping if specified), and temporal trimming.

  Args:
    path: The path from which video will be read.
    output_dir: The directory where the video will be written.
    output_file: The filename as which the video will be written.
    target_fps: Try to downsample frames to achieve this framerate (optional).
    crop: Coords for spatial cropping (x0, y0, x1, y1) (optional).
    scale: Size(s) for spatial scaling. Scalar or (width, height) (optional).
    trim: Frame numbers for temporal trimming (start, end) (optional).
    pix_fmt: Pixel format ('yuv420p', 'yuv444p', etc.)
    codec: The codec to use ('h264', 'h265', 'mjpeg', or 'ffv1')
    crf: Constant rate factor for H.264 encoding (higher = more compression).
    preserve_aspect_ratio: Preserve the aspect ratio if scaling.
    scale_algorithm: The algorithm used for scaling. Default: bicubic
    overwrite: Overwrite file if it exists?
    quiet: Whether to suppress ffmpeg log
    preset: Encoding speed preset (faster encoding vs. compression efficiency)
    hwaccel: Pass 'nvidia' to enable hardware acceleration for h264 or h265
  """
  # Get metadata of original video
  fps, n, w, h, _, _, r, _ = probe_video(path=path)
  # Input
  stream = _ffmpeg_input_from_path(path=path, fps=fps, trim=trim)
  # Filtering
  stream, _, _, _, _ = _ffmpeg_filtering(
    stream=stream, fps=fps, n=n, w=w, h=h, target_fps=target_fps, crop=crop, scale=scale,
    trim=trim, preserve_aspect_ratio=preserve_aspect_ratio, scale_algorithm=scale_algorithm,
    requires_even_dims=(codec=='h264' or codec=='h265'))
  # Output
  _ffmpeg_output_to_file(
    stream, output_dir=output_dir, output_file=output_file, pix_fmt=pix_fmt, codec=codec,
    crf=crf, overwrite=overwrite, quiet=quiet, preset=preset, hwaccel=hwaccel)

def write_video_from_numpy(
    data: np.ndarray,
    fps: Union[float, int],
    pix_fmt: str,
    output_dir: str,
    output_file: str,
    out_pix_fmt: str = 'yuv420p',
    codec: str = 'h264',
    crf: int = 12,
    overwrite: bool = False,
    quiet: bool = True,
    preset: str = 'medium',
    hwaccel: Union[None, str] = None
  ):
  """Write data from a numpy array to a video file.

  Args:
    data: The numpy array with video frames. Shape (n, h, w, 3)
    fps: The frame rate
    pix_fmt: The pixel format of `data`
    output_dir: The directory where the video will be written
    output_file: The filename as which the video will be written
    out_pix_fmt: The pixel format for the output video
    codec: The codec to use ('h264', 'h265', 'mjpeg', or 'ffv1')
    crf: Constant rate factor for H.264 encoding (higher = more compression)
    overwrite: Overwrite if file exists?
    quiet: Suppress ffmpeg output?
    preset: Encoding speed preset (faster encoding vs. compression efficiency)
    hwaccel: Pass 'nvidia' to enable hardware acceleration for h264 or h265
  """
  assert isinstance(data, np.ndarray)
  _, h, w, _ = data.shape
  stream = _ffmpeg_input_from_numpy(w=w, h=h, fps=fps, pix_fmt=pix_fmt)
  buffer = data.flatten().tobytes()
  if (h % 2 != 0 or w % 2 != 0) and (codec=='h264' or codec=='h265'):
    raise ValueError(f"H264 requires both height and width of the video to be even numbers, but received h={h} w={w}")
  _ffmpeg_output_to_file(
    stream, output_dir=output_dir, output_file=output_file, from_stdin=buffer, codec=codec,
    crf=crf, pix_fmt=out_pix_fmt, overwrite=overwrite, quiet=quiet, preset=preset, hwaccel=hwaccel)
