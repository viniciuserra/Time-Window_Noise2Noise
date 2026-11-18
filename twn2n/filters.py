"""Classical filters used as baselines in the paper.

Unlike TWN2N, each of these filters needs information supplied by the user: a
cutoff frequency, a window length, or the noise level.
"""
import numpy as np
from scipy import signal

from ._records import as_records


def lowpass(signals, cutoff_hz, fs, order=4):
    """Zero-phase Butterworth low-pass filter.

    The filter is applied forward and backward (``scipy.signal.sosfiltfilt``),
    so it introduces no phase delay and its attenuation at ``cutoff_hz`` is
    twice that of a single pass (about 6 dB).

    Parameters
    ----------
    signals : array-like or list of array-like
        Record ``(n_samples, n_channels)`` or list of records.
    cutoff_hz : float
        Cutoff frequency, in Hz. In the paper it is fixed at 1.05 times the
        highest natural frequency of the structure.
    fs : float
        Sampling frequency, in Hz.
    order : int, default 4
    """
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    records, restore = as_records(signals)
    return restore([signal.sosfiltfilt(sos, r, axis=0) for r in records])


def savitzky_golay(signals, window=5, polyorder=3):
    """Savitzky-Golay smoothing filter (local polynomial fit).

    Parameters
    ----------
    signals : array-like or list of array-like
    window : int, default 5
        Window length, in samples (odd).
    polyorder : int, default 3
        Order of the fitted polynomial.
    """
    records, restore = as_records(signals)
    return restore([signal.savgol_filter(r, window, polyorder, axis=0) for r in records])


def visushrink(signals, sigma=None, wavelet="sym8", channels=None):
    """Wavelet denoising with the universal threshold (VisuShrink).

    The detail coefficients of the discrete wavelet transform are
    soft-thresholded with ``sigma * sqrt(2 * ln(n_samples))``. It requires the
    optional dependency PyWavelets.

    Parameters
    ----------
    signals : array-like or list of array-like
    sigma : float or None
        Standard deviation of the noise. The paper gives the exact value to the
        filter, which is unavailable in practice. ``None`` estimates it for each
        channel from the finest detail coefficients (median absolute value
        divided by 0.6745).
    wavelet : str, default "sym8"
    channels : sequence of int or None
        Channels that are filtered; the others are returned unchanged.
    """
    try:
        import pywt
    except ImportError as exc:                                      # pragma: no cover
        raise ImportError("visushrink() needs PyWavelets: pip install PyWavelets") from exc

    records, restore = as_records(signals)
    filtered = []
    for record in records:
        out = record.copy()
        n = record.shape[0]
        chosen = range(record.shape[1]) if channels is None else channels
        for ch in chosen:
            coeffs = pywt.wavedec(record[:, ch], wavelet=wavelet)
            level = np.median(np.abs(coeffs[-1])) / 0.6745 if sigma is None else sigma
            threshold = level * np.sqrt(2 * np.log(n))
            coeffs = [coeffs[0]] + [pywt.threshold(c, value=threshold, mode="soft") for c in coeffs[1:]]
            out[:, ch] = pywt.waverec(coeffs, wavelet=wavelet)[:n]
        filtered.append(out)
    return restore(filtered)
