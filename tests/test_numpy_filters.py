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

from prpy.numpy.filters import moving_average, moving_average_size_for_response, moving_std
from prpy.numpy.filters import detrend, detrend_frequency_response, butter_bandpass

import numpy as np
import pytest

MA_ARR = np.array([
  [0.,  .4,  1.2,  2.,   .2 ],
  [0.,  .1,   .5,  .3,  -.1 ],
  [.3,  0.,   .3, -.2,  -.6 ],
  [.2, -.1, -1.2, -.4,   .2 ]
])

MA_REF_AXIS_LAST = np.array([
  [.133333333, .533333333, 1.2, 1.133333333, 0.8],
  [.033333333, .2,         .3,  .233333333, 0.033333333],
  [.2,         .2,         .033333333, -0.166666667, -0.466666667],
  [.1,        -0.366666667,-0.566666667,-0.466666667, 0.]
])

MA_REF_AXIS_0 = np.array([
  [0., .3, .966666667, 1.433333333, 0.1],
  [.1, .166666667, .666666667, .7,  -0.166666667],
  [.166666667, 0., -0.133333333, -0.1, -0.166666667],
  [.233333334, -0.066666667, -0.7, -0.333333333, -0.066666667]
])

MA_REF_AXIS_LAST_LEFT = np.array([
  [.133333333, .133333333, .533333333, 1.2,         1.133333333],
  [.033333333, .033333333, .2,         .3,          .233333333],
  [.2,         .2,         .2,         .033333333, -0.166666667],
  [.1,         .1,        -0.366666667,-0.566666667,-0.466666667]
])

MA_REF_AXIS_0_LEFT = np.array([
  [0., .3, .966666667, 1.433333333, 0.1],
  [0., .3, .966666667, 1.433333333, 0.1],
  [.1, .166666667, .666666667, .7,  -0.166666667],
  [.166666667, 0., -0.133333333, -0.1, -0.166666667]
])

@pytest.mark.parametrize(
  "x,size,axis,center,mode,precision,expected,atol",
  [
    (MA_ARR, 3, -1, True, "reflect", np.float64, MA_REF_AXIS_LAST, 1e-9),
    (MA_ARR, 3, 0,  True, "reflect", np.float64, MA_REF_AXIS_0, 1e-9),
    (MA_ARR, 3, -1, False, "reflect", np.float64, MA_REF_AXIS_LAST_LEFT, 1e-9),
    (MA_ARR, 3,  0, False, "reflect", np.float64, MA_REF_AXIS_0_LEFT, 1e-9),
    (MA_ARR / 1e6, 3, -1, True, "reflect", np.float64, MA_REF_AXIS_LAST / 1e6, 1e-6),
  ]
)
def test_moving_average(x, size, axis, center, mode, precision, expected, atol):
  """Single parametrised test that covers all expected behaviours."""
  x_copy = x.copy()

  out = moving_average(
    x=x,
    size=size,
    axis=axis,
    center=center,
    mode=mode,
    precision=precision
  )

  np.testing.assert_allclose(out, expected, atol=atol)
  np.testing.assert_equal(x, x_copy)
  assert out.dtype == x.dtype

@pytest.mark.parametrize("cutoff_freq", [0.01, 0.1, 1, 10])
def test_moving_average_size_for_response(cutoff_freq):
  # Check that result >= 1
  assert moving_average_size_for_response(sampling_freq=1, cutoff_freq=cutoff_freq) >= 1

def test_moving_std():
  # Check a default use case
  x = np.array([0., .4, 1.2, 2., 2.1, 1.7, 11.4, 4.5, 1.9, 7.6, 6.3, 6.5])
  x_copy = x.copy()
  np.testing.assert_allclose(
    moving_std(x=x,
               size=5, overlap=3, fill_method='mean'),
    np.array([.83809307, .83809307, 2.35538328, 2.35538328, 2.79779145, 3.77764063, 3.57559311, 3.42705292, np.nan, np.nan, np.nan, np.nan]))
  # No side effects
  np.testing.assert_equal(x, x_copy)

def test_detrend():
  # Check a default use case with axis=-1
  z = np.array([[0., .4, 1.2, 2., .2],
                [0., .1, .5, .3, -.1],
                [.3, 0., .3, -.2, -.6],
                [.2, -.1, -1.2, -.4, .2]])
  z_copy = z.copy()
  np.testing.assert_allclose(
    detrend(z=z, Lambda=3, axis=-1),
    np.array([[-.29326743, -.18156859, .36271552, .99234445, -.88022395],
              [-.13389946, -.06970109, .309375, .12595109, -.23172554],
              [-.04370549, -.16474419, .31907328, .03090798, -.14153158],
              [.33796383, .15424475, -.86702586, -.08053786, .45535513]]),
    rtol=1e-6)
  # No side effects
  np.testing.assert_equal(z, z_copy)
  # Check a default use case with axis=0
  np.testing.assert_allclose(
    detrend(z=np.array([[0., .4, 1.2, 2., .2],
                        [0., .1, .5, .3, -.1],
                        [.3, 0., .3, -.2, -.6],
                        [.2, -.1, -1.2, -.4, .2]]), Lambda=3, axis=0),
    np.array([[.01093117, .05725853, -0.1004627, .39976865, .18635049],
              [-0.08016194, -.07703875, -.07755928, -.48877964, -.03799884],
              [.12753036, -.01769809, .45650665, -.22174667, -.48305379],
              [-.0582996, .03747831, -.27848467, .31075766, .33470214]]),
    rtol=1e-6)

def test_detrend_frequency_response():
  # Example from the paper
  f_s = 4
  f_expected = 0.043
  f = detrend_frequency_response(size=800,
                                 Lambda=300,
                                 f_s=f_s)
  np.testing.assert_allclose(f, f_expected, atol=0.01)

def test_butter_bandpass_is_zero_phase():
  fs = 100 # Sampling frequency
  n = 501 # Number of samples (odd to have a perfect center)
  # Create a signal with a single spike exactly in the middle
  signal_in = np.zeros(n)
  center_index = n // 2
  signal_in[center_index] = 1.0
  # Apply the bandpass filter
  signal_out = butter_bandpass(
    x=signal_in, 
    lowcut=5, 
    highcut=20, 
    fs=fs, 
    order=4
  )
  # Find the location of the peak energy in the output
  peak_index_out = np.argmax(np.abs(signal_out))
  # Assert that the peak has not shifted
  # A causal filter like lfilter would fail this test.
  assert center_index == peak_index_out

def test_butter_bandpass_attenuates_and_passes_correctly():
  fs = 500.0       # Sampling frequency
  lowcut = 40.0    # Lower cutoff
  highcut = 60.0   # Upper cutoff
  duration = 2.0   # seconds
  t = np.linspace(0., duration, int(fs * duration), endpoint=False)
  # Create three signals: below, within, and above the passband
  signal_low = np.sin(2 * np.pi * (lowcut / 4) * t)  # 10 Hz
  signal_in_band = np.sin(2 * np.pi * ((lowcut + highcut) / 2) * t) # 50 Hz
  signal_high = np.sin(2 * np.pi * (highcut * 4) * t) # 240 Hz
  power_in_band_original = np.std(signal_in_band)
  signal_in = signal_low + signal_in_band + signal_high
  # Filter the combined signal
  signal_out = butter_bandpass(
    x=signal_in, 
    lowcut=lowcut, 
    highcut=highcut, 
    fs=fs, 
    order=6
  )
  # Analyze the output
  power_out = np.std(signal_out)
  np.testing.assert_allclose(power_out, power_in_band_original, rtol=1e-2)
