###############################################################################
# Copyright (C) Philipp Rouast - All Rights Reserved                          #
# Unauthorized copying of this file, via any medium is strictly prohibited    #
# Proprietary and confidential                                                #
# Written by Philipp Rouast <philipp@rouast.com>, December 2022               #
###############################################################################

import ffmpeg
import itertools
import logging

def find_factors_near(i, f1, f2, f3, s1, s2, s3):
  ts = [(t1, t2, t3) for t1, t2, t3 in list(itertools.product( \
    range(f1-s1, f1+s1+1), range(f2-s2, f2+s2+1), range(f3-s3, f3+s3+1)))]
  for t1, t2, t3 in ts:
    if t1 * t2 * t3 == i:
      return t1, t2, t3
  logging.error("Total={}; Failed to find factors near f1={} f2={} f3={}".format(i, f1, f2, f3))
  raise RuntimeError("Could not find factors near the provided values")

def create_test_video_stream(t):
  """ffmpeg -f lavfi -i testsrc -t 30 -pix_fmt yuv420p testsrc.mp4"""
  stream = ffmpeg.input('testsrc', f='lavfi', t=t)
  return stream
