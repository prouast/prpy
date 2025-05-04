import math
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from scipy.interpolate import PchipInterpolator

def interpolate_filtered(
    x: np.ndarray,
    y: np.ndarray,
    xs: np.ndarray,
    *,
    band: str = "ecg",     # "ecg" | "ppg" | "resp"
    order: int = 4,
    axis: int = 0
) -> np.ndarray:
  """
  Drop‑in replacement for cubic‑spline down/up‑sampling with proper filtering.

  Args
  ----
  x, y, xs : same meaning as before (time grid in seconds + values).
  band     : choose pre‑set pass‑band; or pass a (low, high) tuple in Hz.
  order    : Butterworth order.
  axis     : time axis of y.
  """
  # numpyfy
  x, y, xs = map(np.asarray, (x, y, xs))

  # Handle NaNs cleanly (don’t turn them into zeros)
  mask = np.isnan(y)
  if mask.any():
    y = y.copy()
    y[mask] = np.interp(x[mask], x[~mask], y[~mask])

  # If grids identical, nothing to do
  if np.array_equal(x, xs):
    return y

  # --- 1. Decide pass‑band
  presets = {
    "ecg":  (0.5, 40.0),
    "ppg":  (0.5, 8.0),
    "resp": (0.1, 0.5),
  }
  if isinstance(band, str):
    if band not in presets:
      raise ValueError(f"band must be one of {list(presets)} or a (low, high) tuple")
    low, high = presets[band]
  else:
    low, high = band

  # --- 2. Check if x and xs are (almost) uniform
  def is_uniform(t, tol=1e-6):
    dt = np.diff(t)
    return np.allclose(dt, dt[0], rtol=0, atol=tol)

  uniform_in  = is_uniform(x)
  uniform_out = is_uniform(xs)

  if uniform_in and uniform_out:
    # Fast path: true resampling ---------------------------------------------
    fs_in  = 1.0 / np.median(np.diff(x))
    fs_out = 1.0 / np.median(np.diff(xs))

    # Design zero‑phase band‑pass
    nyq = 0.5 * fs_in
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    y_filt = filtfilt(b, a, y, axis=axis)

    # Poly‑phase resample
    gcd = math.gcd(int(round(fs_in)), int(round(fs_out)))
    up   = int(round(fs_out)) // gcd
    down = int(round(fs_in))  // gcd
    y_rs = resample_poly(y_filt, up, down, axis=axis)

    # Align length exactly to len(xs)
    if y_rs.shape[axis] != xs.size:
      slicer = [slice(None)] * y_rs.ndim
      slicer[axis] = slice(0, xs.size)
      y_rs = y_rs[tuple(slicer)]
    return y_rs
  else:
    # Fallback: shape‑preserving spline (PCHIP) -------------------------------
    pchip = PchipInterpolator(x, y, axis=axis, extrapolate=False)
    return pchip(xs)
