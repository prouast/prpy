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
import itertools
import logging
from typing import Tuple

def find_factors_near(
    i: int,
    f1: int,
    f2: int,
    f3: int,
    max_delta_1: int,
    max_delta_2: int,
    max_delta_3: int
  ) -> Tuple[int, int, int]:
  """Search for factors of a number near provided approximate factors.
  Used to determine dim of filtered video which can minimally deviate from targets

  Args:
    i: The number to be factorized
    f1: The first factor to search nearby
    f2: The second factor to search nearby
    f3: The third factor to search nearby
    max_delta_1: Maximum deviation allowed for f1
    max_delta_2: Maximum deviation allowed for f2
    max_delta_3: Maximum deviation allowed for f3
  Returns:
    Tuple of
     - f1: The actual first factor
     - f2: The actual second factor
     - f3: The actual third factor
  """
  assert isinstance(i, int)
  assert isinstance(f1, int)
  assert isinstance(f2, int)
  assert isinstance(f3, int)
  assert isinstance(max_delta_1, int)
  assert isinstance(max_delta_2, int)
  assert isinstance(max_delta_3, int)
  # Iterative deepening
  for delta in range(max(max_delta_1, max_delta_2, max_delta_3)+1):
    delta_1 = min(delta, max_delta_1)
    delta_2 = min(delta, max_delta_2)
    delta_3 = min(delta, max_delta_3)
    ts = [(t1, t2, t3) for t1, t2, t3 in list(itertools.product( \
      range(f1-delta_1, f1+delta_1+1), range(f2-delta_2, f2+delta_2+1), range(f3-delta_3, f3+delta_3+1)))]
    for t1, t2, t3 in ts:
      if t1 * t2 * t3 == i:
        return t1, t2, t3
  logging.error("Total={}; Failed to find factors near f1={} f2={} f3={} at delta=({}, {}, {})".format(i, f1, f2, f3, max_delta_1, max_delta_2, max_delta_3))
  raise RuntimeError("Could not find factors near the provided values")

def create_test_video_stream(t: int) -> ffmpeg.nodes.FilterableStream:
  """Create an ffmpeg video stream for testing.
  Like `ffmpeg -f lavfi -i testsrc -t 30 -pix_fmt yuv420p testsrc.mp4`
  
  Args:
    t: The test stream time in seconds
  Returns:
    stream: The test stream
  """
  stream = ffmpeg.input('testsrc', f='lavfi', t=t)
  return stream
