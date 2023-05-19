###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import ffmpeg
import itertools
import logging

def find_factors_near(i, f1, f2, f3, max_delta_1, max_delta_2, max_delta_3):
  # Iterative deepening
  for delta in range(max(max_delta_1, max_delta_2, max_delta_3)):
    delta_1 = min(delta, max_delta_1)
    delta_2 = min(delta, max_delta_2)
    delta_3 = min(delta, max_delta_3)
    ts = [(t1, t2, t3) for t1, t2, t3 in list(itertools.product( \
      range(f1-delta_1, f1+delta_1+1), range(f2-delta_2, f2+delta_2+1), range(f3-delta_3, f3+delta_3+1)))]
    for t1, t2, t3 in ts:
      if t1 * t2 * t3 == i:
        return t1, t2, t3
  logging.error("Total={}; Failed to find factors near f1={} f2={} f3={}".format(i, f1, f2, f3))
  raise RuntimeError("Could not find factors near the provided values")

def create_test_video_stream(t):
  """ffmpeg -f lavfi -i testsrc -t 30 -pix_fmt yuv420p testsrc.mp4"""
  stream = ffmpeg.input('testsrc', f='lavfi', t=t)
  return stream
