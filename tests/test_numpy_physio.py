# Copyright (c) 2025 Philipp Rouast
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

import sys
sys.path.append('../prpy')

from prpy.constants import SECONDS_PER_MINUTE
from prpy.numpy.filters import moving_average
from prpy.numpy.physio import EMethod, EScope, HR_MIN, HR_MAX
from prpy.numpy.physio import estimate_rate_per_minute_from_signal

import numpy as np
import pytest

@pytest.mark.parametrize("signal", ["ecg", "ppg"])
@pytest.mark.parametrize("method", [EMethod.PEAK, EMethod.PERIODOGRAM])
def test_estimate_rate_per_minute_from_signal_global(synthetic_ecg_static, synthetic_ppg_static, signal, method):
  s = synthetic_ecg_static if signal == "ecg" else synthetic_ppg_static
  rate = estimate_rate_per_minute_from_signal(signal=s,
                                              f_s=250.,
                                              f_range=(HR_MIN/SECONDS_PER_MINUTE, HR_MAX/SECONDS_PER_MINUTE),
                                              scope=EScope.GLOBAL,
                                              method=method,
                                              window_size=2000,
                                              interp_skipped=False,
                                              height=1. if signal == 'ecg' and method == EMethod.PEAK else 0.)
  assert rate == 60.

@pytest.mark.parametrize("signal", ["ecg", "ppg"])
@pytest.mark.parametrize("interp_skipped", [False, True])
@pytest.mark.parametrize("method", [EMethod.PEAK, EMethod.PERIODOGRAM])
def test_estimate_rate_per_minute_from_signal_rolling_on_static_signal(synthetic_ecg_static, synthetic_ppg_static, signal, method, interp_skipped):
  s = synthetic_ecg_static if signal == "ecg" else synthetic_ppg_static
  if signal == 'ecg' and method == EMethod.PERIODOGRAM:
    s = moving_average(s, size=101)
  out = estimate_rate_per_minute_from_signal(signal=s,
                                             f_s=250.,
                                             f_range=(HR_MIN/SECONDS_PER_MINUTE, HR_MAX/SECONDS_PER_MINUTE),
                                             scope=EScope.ROLLING,
                                             method=method,
                                             window_size=2500,
                                             overlap=2000,
                                             interp_skipped=interp_skipped,
                                             height=1. if signal == 'ecg' and method == EMethod.PEAK else 0.)
  assert out.shape == s.shape
  np.testing.assert_allclose(out, 60., atol=1.)

@pytest.mark.parametrize("signal", ["ecg", "ppg"])
@pytest.mark.parametrize("method", [EMethod.PEAK, EMethod.PERIODOGRAM])
def test_estimate_rate_per_minute_from_signal_rolling_on_dynamic_signal(synthetic_ecg_dynamic, synthetic_ppg_dynamic, signal, method):
  s = synthetic_ecg_dynamic if signal == "ecg" else synthetic_ppg_dynamic
  out = estimate_rate_per_minute_from_signal(signal=s,
                                             f_s=250.,
                                             f_range=(HR_MIN/SECONDS_PER_MINUTE, HR_MAX/SECONDS_PER_MINUTE),
                                             scope=EScope.ROLLING,
                                             method=method,
                                             window_size=2500,
                                             overlap=2000,
                                             interp_skipped=True,
                                             height=1. if signal == 'ecg' and method == EMethod.PEAK else 0.)
  assert out.shape == s.shape
  np.testing.assert_allclose(out, 66., atol=8.)
  assert np.all(np.diff(out) >= 0)
  assert out[-1] - out[0] > 10
  