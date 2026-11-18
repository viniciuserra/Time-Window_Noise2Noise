"""Signal-to-noise ratio (SNR) metrics.

The definitions are those of the paper. With a clean reference ``x0`` and an
estimate ``x`` (the noisy measurement or the output of a denoiser):

    SNR = 20 * log10( std(x0) / std(x - x0) )   [dB]
"""
import numpy as np
from scipy import signal

from ._records import as_records, select_channels


def snr_db(clean, estimate, channels=None, per_channel=False):
    """SNR of ``estimate`` with respect to the clean reference, in dB.

    Parameters
    ----------
    clean, estimate : array-like or list of array-like
        Record ``(n_samples, n_channels)`` or list of records, with the same
        shapes.
    channels : sequence of int or None
        Channels included in the metric. ``None`` means all of them.
    per_channel : bool, default False
        If ``False``, the selected channels and all records are pooled into a
        single value, as in the tables of the paper. Pool only channels with
        the same physical unit (for example, the displacements). If ``True``,
        one value per channel is returned.

    Returns
    -------
    snr : float or numpy.ndarray
    """
    c, e = _pooled(clean, channels), _pooled(estimate, channels)
    if c.shape != e.shape:
        raise ValueError(f"clean and estimate have different shapes: {c.shape} and {e.shape}.")
    if per_channel:
        return 20 * np.log10(np.std(c, axis=0) / np.std(e - c, axis=0))
    return float(20 * np.log10(np.std(c) / np.std(e - c)))


def snr_gain_db(clean, noisy, denoised, channels=None, per_channel=False):
    """SNR gain of a denoiser: output SNR minus input SNR, in dB."""
    out = snr_db(clean, denoised, channels, per_channel)
    return out - snr_db(clean, noisy, channels, per_channel)


def band_snr_db(clean, estimate, band, fs, channels=None):
    """SNR restricted to a frequency band, in dB.

    The energy of the clean signals inside ``band`` is divided by the energy of
    the error inside the same band. Spectra are computed per channel and per
    record with a Tukey window (5% taper) and the energies are summed over the
    selected channels and over the records.

    Parameters
    ----------
    clean, estimate : array-like or list of array-like
    band : (float, float)
        Lower and upper limits of the band, in Hz.
    fs : float
        Sampling frequency, in Hz.
    channels : sequence of int or None
    """
    num = den = 0.0
    for c, e in _channel_pairs(clean, estimate, channels):
        f = np.fft.rfftfreq(len(c), 1.0 / fs)
        mask = (f >= band[0]) & (f <= band[1])
        spec_c, spec_e = _tukey_rfft(c)[mask], _tukey_rfft(e)[mask]
        num += np.sum(np.abs(spec_c) ** 2)
        den += np.sum(np.abs(spec_c - spec_e) ** 2)
    return float(10 * np.log10(num / den))


def band_noise_rejection_db(clean, noisy, estimate, band, fs, channels=None):
    """Reduction of the error energy inside a band, from the noisy input to the estimate, in dB."""
    e_in = e_out = 0.0
    records_c, _ = as_records(clean)
    records_n, _ = as_records(noisy)
    records_e, _ = as_records(estimate)
    for rc, rn, re in zip(records_c, records_n, records_e):
        rc, rn, re = (select_channels(r, channels) for r in (rc, rn, re))
        for ch in range(rc.shape[1]):
            f = np.fft.rfftfreq(rc.shape[0], 1.0 / fs)
            mask = (f >= band[0]) & (f <= band[1])
            e_in += np.sum(np.abs(_tukey_rfft(rn[:, ch] - rc[:, ch])[mask]) ** 2)
            e_out += np.sum(np.abs(_tukey_rfft(re[:, ch] - rc[:, ch])[mask]) ** 2)
    return float(10 * np.log10(e_in / e_out))


def _pooled(signals, channels):
    records, _ = as_records(signals)
    return np.vstack([select_channels(r, channels) for r in records])


def _channel_pairs(clean, estimate, channels):
    records_c, _ = as_records(clean)
    records_e, _ = as_records(estimate)
    if len(records_c) != len(records_e):
        raise ValueError("clean and estimate have a different number of records.")
    for rc, re in zip(records_c, records_e):
        rc, re = select_channels(rc, channels), select_channels(re, channels)
        if rc.shape != re.shape:
            raise ValueError(f"clean and estimate have different shapes: {rc.shape} and {re.shape}.")
        for ch in range(rc.shape[1]):
            yield rc[:, ch], re[:, ch]


def _tukey_rfft(x, alpha=0.05):
    return np.fft.rfft(x * signal.windows.tukey(len(x), alpha))
