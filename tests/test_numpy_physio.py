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
from prpy.numpy.filters import moving_average, detrend, detrend_frequency_response
from prpy.numpy.freq import estimate_freq
from prpy.numpy.physio import EMethod, EScope, EWindowUnit, HR_MIN, HR_MAX, HRVMetric
from prpy.numpy.physio import estimate_rate_from_signal
from prpy.numpy.physio import estimate_rate_from_detections
from prpy.numpy.physio import estimate_rate_from_detection_sequences
from prpy.numpy.physio import estimate_hrv_from_signal
from prpy.numpy.physio import estimate_hrv_from_detections
from prpy.numpy.physio import estimate_hrv_from_detection_sequences
from prpy.numpy.physio import moving_average_size_for_hr_response, moving_average_size_for_rr_response
from prpy.numpy.physio import detrend_lambda_for_hr_response, detrend_lambda_for_rr_response

import numpy as np
import pytest

@pytest.mark.parametrize("signal", ["ecg", "ppg"])
@pytest.mark.parametrize("method", [EMethod.PEAK, EMethod.PERIODOGRAM])
def test_estimate_rate_from_signal_global(synthetic_ecg_static, synthetic_ppg_static, signal, method):
  s = synthetic_ecg_static if signal == "ecg" else synthetic_ppg_static
  rate = estimate_rate_from_signal(signal=s,
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
def test_estimate_rate_from_signal_rolling_on_static_signal(synthetic_ecg_static, synthetic_ppg_static, signal, method, interp_skipped):
  s = synthetic_ecg_static if signal == "ecg" else synthetic_ppg_static
  if signal == 'ecg' and method == EMethod.PERIODOGRAM:
    s = moving_average(s, size=101)
  out = estimate_rate_from_signal(signal=s,
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
def test_estimate_rate_from_signal_rolling_on_dynamic_signal(synthetic_ecg_dynamic, synthetic_ppg_dynamic, signal, method):
  s = synthetic_ecg_dynamic if signal == "ecg" else synthetic_ppg_dynamic
  out = estimate_rate_from_signal(signal=s,
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

def make_dets_static(hr_bpm: float, n_beats: int = 20, f_s: float = 250.):
  """Evenly spaced detection indices for a constant HR."""
  rr = f_s * SECONDS_PER_MINUTE / hr_bpm
  dets = (np.arange(n_beats) * rr).round().astype(int)
  t = np.arange(dets[-1] + 1, dtype=float) / f_s
  return dets, t

def make_dets_dynamic(hr_start: float, hr_end: float, n_beats: int = 30, f_s: float = 250.):
  """Linear HR ramp."""
  rates = np.linspace(hr_start, hr_end, n_beats)
  rr = f_s * SECONDS_PER_MINUTE / rates
  dets = np.cumsum(np.insert(rr[:-1], 0, 0)).round().astype(int)
  t = np.arange(dets[-1] + 1, dtype=float) / f_s
  return dets, t

@pytest.mark.parametrize("hr", [50, 80, 120])
def test_estimate_rate_from_detections_global_static(hr):
  f_s = 250.
  n_beats = 20
  dets, _ = make_dets_static(hr_bpm=hr, n_beats=n_beats, f_s=f_s)
  est = estimate_rate_from_detections(
    dets, f_s=f_s, scope=EScope.GLOBAL
  )
  assert pytest.approx(est, rel=0.01) == hr

@pytest.mark.parametrize("window", [5, 10])
def test_estimate_rate_from_detections_rolling_static(window):
  f_s = 250.
  hr = 75
  dets, t = make_dets_static(hr_bpm=hr, n_beats=40, f_s=f_s)
  out = estimate_rate_from_detections(
    dets,
    f_s=f_s,
    t=t,
    scope=EScope.ROLLING,
    window_unit=EWindowUnit.DETECTIONS,
    min_window_size=window,
    max_window_size=window,
    overlap=window - 1,
  )
  assert out.shape == t.shape
  finite = np.isfinite(out)
  assert np.allclose(out[finite], 75, atol=0.8)

def test_estimate_rate_from_detections_rolling_dynamic():
  f_s = 250.
  dets, t = make_dets_dynamic(hr_start=60, hr_end=90, n_beats=30, f_s=f_s)
  out = estimate_rate_from_detections(
    dets,
    f_s=f_s,
    t=t,
    scope=EScope.ROLLING,
    window_unit=EWindowUnit.DETECTIONS,
    min_window_size=6,
    max_window_size=6,
    overlap=5,
    interp_skipped=True,
  )
  finite = out[np.isfinite(out)]
  assert finite.size >= 1
  assert np.all(np.diff(finite) >= -1e-6)
  assert finite[-1] - finite[0] > 20

@pytest.mark.parametrize("window_s", [1.0, 2.0])
def test_estimate_rate_from_detections_rolling_static_seconds(window_s):
  f_s = 250.
  hr = 75
  dets, t = make_dets_static(hr_bpm=hr, n_beats=40, f_s=f_s)
  out = estimate_rate_from_detections(
    dets,
    f_s=f_s,
    t=t,
    scope=EScope.ROLLING,
    window_unit=EWindowUnit.SECONDS,
    min_window_size=window_s,
    max_window_size=window_s,
  )
  assert out.shape == t.shape
  finite = np.isfinite(out)
  assert finite.sum() > 0
  assert np.allclose(out[finite], hr, atol=0.8)

def test_estimate_rate_from_detections_rolling_dynamic_seconds():
  f_s = 250.
  dets, t = make_dets_dynamic(hr_start=60, hr_end=90, n_beats=30, f_s=f_s)
  out = estimate_rate_from_detections(
    dets,
    f_s=f_s,
    t=t,
    scope=EScope.ROLLING,
    window_unit=EWindowUnit.SECONDS,
    min_window_size=3.0,
    max_window_size=3.0,
    interp_skipped=True,
  )
  assert out.shape == t.shape
  finite = out[np.isfinite(out)]
  assert finite.size > 0
  assert np.all(np.diff(finite) >= -1e-6)
  assert finite[-1] - finite[0] > 20

def _two_static_runs(hr_bpm: float, gap_s: float = 2.0, *, f_s: float = 250.):
  """Return two equal-HR detection runs separated by `gap_s` seconds."""
  rr = f_s * SECONDS_PER_MINUTE / hr_bpm
  k = 12
  seq1 = (np.arange(k) * rr).round().astype(int)
  shift = int(seq1[-1] + gap_s * f_s + rr)
  seq2 = seq1 + shift
  dets_all = np.hstack([seq1, seq2])
  t = np.arange(dets_all[-1] + 1, dtype=float) / f_s
  return [seq1, seq2], t

def _two_dynamic_runs(hr0: float, hr1: float, *, f_s: float = 250.):
  """Rising HR split into two runs of equal length."""
  k = 30
  dets, t_all = make_dets_dynamic(hr_start=hr0, hr_end=hr1, n_beats=k, f_s=f_s)
  seq1, seq2 = np.array_split(dets, 2)
  return [seq1, seq2], t_all

@pytest.mark.parametrize("hr", [55, 90, 130])
def test_estimate_rate_from_detection_sequences_global_static(hr):
  f_s = 250.
  seqs, _ = _two_static_runs(hr_bpm=hr, f_s=f_s)
  est = estimate_rate_from_detection_sequences(
    seqs, f_s=f_s, scope=EScope.GLOBAL
  )
  assert pytest.approx(est, rel=0.01) == hr

@pytest.mark.parametrize("window", [5, 8])
def test_estimate_rate_from_detection_sequences_rolling_static_detections(window):
  f_s = 250.
  hr = 75
  seqs, t = _two_static_runs(hr_bpm=hr, f_s=f_s)
  out = estimate_rate_from_detection_sequences(
    seqs,
    f_s=f_s,
    t=t,
    scope=EScope.ROLLING,
    window_unit=EWindowUnit.DETECTIONS,
    min_window_size=window,
    max_window_size=window,
    overlap=window - 1,
  )
  assert out.shape == t.shape
  finite = np.isfinite(out)
  assert finite.sum() > 0
  np.testing.assert_allclose(out[finite], hr, atol=0.8)

@pytest.mark.parametrize("window_s", [1.0, 2.0])
def test_estimate_rate_from_detection_sequences_rolling_static_seconds(window_s):
  f_s = 250.
  hr = 75
  seqs, t = _two_static_runs(hr_bpm=hr, f_s=f_s)
  out = estimate_rate_from_detection_sequences(
    seqs,
    f_s=f_s,
    t=t,
    scope=EScope.ROLLING,
    window_unit=EWindowUnit.SECONDS,
    min_window_size=window_s,
    max_window_size=window_s,
  )
  assert out.shape == t.shape
  finite = np.isfinite(out)
  assert finite.sum() > 0
  np.testing.assert_allclose(out[finite], hr, atol=0.8)

def test_estimate_rate_from_detection_sequences_rolling_dynamic_detections():
  f_s = 250.
  seqs, t = _two_dynamic_runs(hr0=60, hr1=90, f_s=f_s)
  out = estimate_rate_from_detection_sequences(
    seqs,
    f_s=f_s,
    t=t,
    scope=EScope.ROLLING,
    window_unit=EWindowUnit.DETECTIONS,
    min_window_size=6,
    max_window_size=6,
    overlap=5,
    interp_skipped=True,
  )
  assert out.shape == t.shape
  finite = out[np.isfinite(out)]
  assert finite.size > 0
  assert np.all(np.diff(finite) >= -1e-6)
  assert finite[-1] - finite[0] > 20

def test_estimate_rate_from_detection_sequences_rolling_dynamic_seconds():
  f_s = 250.
  seqs, t = _two_dynamic_runs(hr0=60, hr1=90, f_s=f_s)
  out = estimate_rate_from_detection_sequences(
    seqs,
    f_s=f_s,
    t=t,
    scope=EScope.ROLLING,
    window_unit=EWindowUnit.SECONDS,
    min_window_size=10,
    max_window_size=10,
    interp_skipped=True,
  )
  assert out.shape == t.shape
  finite = out[np.isfinite(out)]
  assert finite.size > 0
  assert np.all(np.diff(finite) >= -1e-6)
  assert finite[-1] - finite[0] > 15

def test_estimate_hrv_sdnn_from_detections_global():
  det_idxs = np.asarray([202, 392, 601, 799, 1201, 1403, 1610, 1839])
  actual = estimate_hrv_from_detections(det_idxs, metric=HRVMetric.SDNN, f_s=30, interp_skipped=True, min_dets=5, min_t=1.)
  np.testing.assert_allclose(200, actual, atol=0.1)

@pytest.mark.parametrize("correct_quantization_error", [False, True])
def test_estimate_hrv_sdnn_from_detection_sequences_global(correct_quantization_error):
  idxs_list = [[202, 392, 612, 799], [1201, 1403, 1610, 1839]]
  t = np.linspace(0, 8, 2000)
  out = estimate_hrv_from_detection_sequences(seqs=idxs_list,
                                                   t=t,
                                                   metric=HRVMetric.SDNN,
                                                   correct_quantization_error=correct_quantization_error,
                                                   scope=EScope.GLOBAL,
                                                   min_dets=2,
                                                   min_t=.5)
  if correct_quantization_error:
    np.testing.assert_allclose(out, 53.0593762, atol=1e-5)
  else:
    np.testing.assert_allclose(out, 53.0721198, atol=1e-5)

@pytest.mark.parametrize("correct_quantization_error", [False, True])
def test_estimate_hrv_sdnn_from_detection_sequences_rolling(correct_quantization_error):
  idxs_list = [[202, 392, 612, 799], [1201, 1403, 1610, 1839]]
  t = np.linspace(0, 8, 2000)
  out = estimate_hrv_from_detection_sequences(seqs=idxs_list,
                                                   t=t,
                                                   metric=HRVMetric.SDNN,
                                                   correct_quantization_error=correct_quantization_error,
                                                   min_window_size=2,
                                                   max_window_size=4,
                                                   scope=EScope.ROLLING,
                                                   overlap=2,
                                                   min_dets=3,
                                                   min_t=.5)
  if correct_quantization_error:
    assert np.isnan(out[0]) # Start
    assert np.isnan(out[202]) # Seq 1 val 1
    assert np.isnan(out[392]) # Seq 1 val 2
    assert np.isnan(out[611]) # Just before seq 1 val 3
    assert pytest.approx(out[612], abs=0.001) == 59.6173 # Seq 1 val 3
    assert pytest.approx(out[798], abs=0.001) == 59.6173 # Just before seq 1 val 4
    assert pytest.approx(out[799], abs=0.001) == 59.6173 # Seq 1 val 4
    assert pytest.approx(out[985], abs=0.001) == 59.6173 # In gap, but allow expected time after det to be valid
    assert np.isnan(out[1201]) # Seq 2 val 1
    assert np.isnan(out[1403]) # Seq 2 val 2
    assert np.isnan(out[1609]) # Just before seq 2 val 3
    assert pytest.approx(out[1610], abs=0.001) == 46.9229 # Seq 2 val 3
    assert pytest.approx(out[1839], abs=0.001) == 46.9229 # Seq 2 val 4
    assert pytest.approx(out[1999], abs=0.001) == 46.9229 # In gap, but allow expected time after det to be valid
  else:
    assert np.isnan(out[0]) # Start
    assert np.isnan(out[202]) # Seq 1 val 1
    assert np.isnan(out[392]) # Seq 1 val 2
    assert np.isnan(out[611]) # Just before seq 1 val 3
    assert pytest.approx(out[612], abs=0.001) == 59.6284 # Seq 1 val 3
    assert pytest.approx(out[798], abs=0.001) == 59.6284 # Just before seq 1 val 4
    assert pytest.approx(out[799], abs=0.001) == 59.6284 # Seq 1 val 4
    assert pytest.approx(out[985], abs=0.001) == 59.6284 # In gap, but allow expected time after det to be valid
    assert np.isnan(out[1201]) # Seq 2 val 1
    assert np.isnan(out[1403]) # Seq 2 val 2
    assert np.isnan(out[1609]) # Just before seq 2 val 3
    assert pytest.approx(out[1610], abs=0.001) == 46.9371 # Seq 2 val 3
    assert pytest.approx(out[1839], abs=0.001) == 46.9371 # Seq 2 val 4
    assert pytest.approx(out[1999], abs=0.001) == 46.9371 # In gap, but allow expected time after det to be valid

def test_moving_average_size_for_hr_response():
  out = moving_average_size_for_hr_response(30.)
  assert out > 1 and out < 5
  assert isinstance(out, int)

def test_moving_average_size_for_rr_response():
  out = moving_average_size_for_rr_response(30.)
  assert out > 10 and out < 15
  assert isinstance(out, int)

def test_detrend_lambda_for_hr_response():
  out = detrend_lambda_for_hr_response(30.)
  assert out > 100 and out < 150
  assert isinstance(out, int)

def test_detrend_lambda_for_rr_response():
  out = detrend_lambda_for_rr_response(30.)
  assert out > 6000 and out < 8000
  assert isinstance(out, int)

@pytest.mark.parametrize("f_s", [30, 125])
def test_detrend_lambda_for_hr_response_preserves_frequency(f_s):
  t = 10
  f_ppg = 40./60.
  # Test data
  num = t * f_s
  x = np.linspace(0, f_ppg * 2 * np.pi * t, num=num)
  np.random.seed(0)
  y_ = 100 * np.sin(x) + np.random.normal(scale=8, size=num) + x
  # Compute detrend lambda
  Lambda = detrend_lambda_for_hr_response(f_s=f_s)
  f_theoretical = detrend_frequency_response(size=num, Lambda=Lambda, f_s=f_s)
  assert f_theoretical < f_ppg
  assert isinstance(Lambda, int)
  assert Lambda > 0
  # Detrend
  y = detrend(y_, Lambda)
  # Make sure frequency is preserved
  np.testing.assert_allclose(
    estimate_freq(x=y, f_s=f_s, f_res=0.01, f_range=(40./60., 240./60.), method='periodogram'),
    f_ppg,
    atol=0.01)

@pytest.mark.parametrize("f_s", [30, 125])
def test_detrend_lambda_for_rr_response_preserves_frequency(f_s):
  t = 20
  f_resp = 7./60.
  # Test data
  num = t * f_s
  x = np.linspace(0, f_resp * 2 * np.pi * t, num=num)
  np.random.seed(0)
  y_ = 100 * np.sin(x) + np.random.normal(scale=8, size=num) + x
  # Compute detrend lambda
  Lambda = detrend_lambda_for_rr_response(f_s=f_s)
  f_theoretical = detrend_frequency_response(size=num, Lambda=Lambda, f_s=f_s)
  assert f_theoretical < f_resp
  assert isinstance(Lambda, int)
  assert Lambda > 0
  # Detrend
  y = detrend(y_, Lambda)
  # Make sure frequency is preserved
  np.testing.assert_allclose(
    estimate_freq(x=y, f_s=f_s, f_res=0.01, f_range=(5./60., 100./60.), method='periodogram'),
    f_resp,
    atol=0.01)
  
@pytest.mark.parametrize("scenario", [([.01, .03, .2,  .6,  .4,  .001, .02, .01, .8, .09, .03, .06, .99, .03, .1, .06, .11, .7, .31, .12, .03, .79, .1, .05, .01, .94, 0.1],
                                       [1.,  1.,  .99, .95, .96, 1.,   1.,  1.,  1., 1.,  1.,  1.,  1.,  1.,  1., 1.,  .81, .5, .9,  1.,  1.,  1.,  1.,  1., 1.,  1., 1.],
                                       4., .6, 93.17, .95)])
def test_estimate_hrv_sdnn_from_signal_with_confidence(scenario):
  signal, conf, f_s, conf_threshold, expected, exp_conf = scenario
  signal = np.asarray(signal)
  conf = np.asarray(conf)
  actual, conf = estimate_hrv_from_signal(
    signal=signal,
    metric=HRVMetric.SDNN,
    f_s=f_s,
    min_window_size=27,
    max_window_size=27,
    overlap=0,
    scope=EScope.GLOBAL,
    confidence=conf,
    confidence_threshold=conf_threshold,
    min_t=.5,
    min_dets=4
  )
  np.testing.assert_allclose(actual, expected, atol=1e-2)
  np.testing.assert_allclose(conf, exp_conf)
