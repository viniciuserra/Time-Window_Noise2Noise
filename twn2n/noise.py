"""Noise models of the paper: white, pink, and brown noise, and quantization.

The functions add noise to clean signals at a prescribed signal-to-noise ratio
(SNR), which is how the benchmarks of the paper are corrupted. They are useful
to test a denoiser on signals whose clean version is known.
"""
import numpy as np

from ._records import as_records

NOISE_KINDS = ("white", "pink", "brown", "quantization")
_ALIASES = {"quant": "quantization", "gaussian": "white", "red": "brown", "brownian": "brown"}


def colored_noise(kind, n_samples, std=1.0, rng=None, exact_std=True):
    """Generate one noise sequence with a prescribed standard deviation.

    Parameters
    ----------
    kind : {"white", "pink", "brown"}
        ``"white"``: independent Gaussian samples (flat spectrum).
        ``"pink"``: Gaussian noise shaped in the frequency domain so that the
        power spectral density decays as ``1 / f``.
        ``"brown"``: cumulative sum of Gaussian samples (random walk), whose
        power spectral density decays as ``1 / f**2``.
    n_samples : int
    std : float, default 1.0
        Standard deviation of the returned sequence.
    rng : numpy.random.Generator, int, or None
        Random generator or seed.
    exact_std : bool, default True
        If ``True``, the sequence is rescaled so that its sample standard
        deviation equals ``std`` exactly. If ``False``, white noise is drawn
        with a nominal standard deviation ``std`` and keeps its sampling
        fluctuation; pink and brown noise are always rescaled, because their
        construction does not fix the scale.

    Returns
    -------
    noise : numpy.ndarray, shape (n_samples,)
    """
    kind = _canonical(kind)
    rng = _generator(rng)
    if kind == "white":
        x = rng.standard_normal(n_samples)
        if not exact_std:
            return std * x
    elif kind == "pink":
        spectrum = np.fft.rfft(rng.standard_normal(n_samples))
        f = np.fft.rfftfreq(n_samples)
        scale = 1.0 / (np.sqrt(f) + 1e-10)
        scale[0] = 0.0
        x = np.fft.irfft(spectrum * scale, n=n_samples)
    elif kind == "brown":
        x = np.cumsum(rng.standard_normal(n_samples))
        x = x - x.mean()
    else:
        raise ValueError("Quantization is not an additive sequence; use quantize() or add_noise().")
    current = np.std(x)
    return x if current < 1e-12 else x * (std / current)


def add_noise(signals, kind="white", snr_db=15.0, rng=None, channels=None, exact_snr=True,
              **quantization):
    """Corrupt clean signals with one of the noise models of the paper.

    For the additive models, an independent sequence is added to every channel
    of every record. Its standard deviation is set from the standard deviation
    of that channel, so that the channel has the requested SNR:

        sigma_noise = sigma_channel / 10 ** (snr_db / 20)

    For ``kind="quantization"`` the signals are rounded by an ideal converter
    (see :func:`quantize`); ``snr_db`` is then ignored, because the noise level
    is set by the resolution of the converter.

    Parameters
    ----------
    signals : array-like or list of array-like
        Clean record ``(n_samples, n_channels)`` or list of records.
    kind : {"white", "pink", "brown", "quantization"}
    snr_db : float, default 15.0
        Target SNR of each channel, in dB.
    rng : numpy.random.Generator, int, or None
        Random generator or seed. Sequences are drawn record by record and,
        inside a record, channel by channel.
    channels : sequence of int or None
        Channels that receive noise. ``None`` means all of them. Channels that
        are constant inside a record are always left untouched.
    exact_snr : bool, default True
        If ``True``, each noise sequence is rescaled to the target standard
        deviation, so every channel has exactly the requested SNR. If
        ``False``, white noise keeps its sampling fluctuation and the SNR is
        the requested one only on average.
    **quantization
        Arguments of :func:`quantize` (``bits``, ``peak_fraction``).

    Returns
    -------
    noisy : same container and shape as ``signals``
    """
    kind = _canonical(kind)
    if kind == "quantization":
        return quantize(signals, channels=channels, **quantization)
    if quantization:
        raise TypeError(f"Unexpected arguments for {kind} noise: {sorted(quantization)}.")
    rng = _generator(rng)
    records, restore = as_records(signals)
    noisy = []
    for record in records:
        out = record.copy()
        for ch in _channel_list(channels, record.shape[1]):
            sigma = np.std(record[:, ch])
            if sigma < 1e-14:
                continue
            out[:, ch] += colored_noise(kind, record.shape[0], sigma / (10 ** (snr_db / 20.0)), rng,
                                        exact_std=exact_snr)
        noisy.append(out)
    return restore(noisy)


def quantize(signals, bits=12, peak_fraction=0.0035, channels=None):
    """Round signals to the levels of an ideal analog-to-digital converter.

    Each channel of each record is scaled so that its peak absolute value
    occupies ``peak_fraction`` of the positive full scale of a converter with
    ``bits`` bits, rounded to the nearest level, and scaled back. A small
    ``peak_fraction`` represents a weak signal acquired with a range that is
    too wide: the signal spans few levels and the rounding error is large.

    With the default values (12 bits and 0.35% of the full scale, that is, a
    peak of 35 mV in a converter of +-10 V) the peak spans about seven levels,
    and the SNR of the displacements of the vibration benchmark is close to
    15 dB.

    Parameters
    ----------
    signals : array-like or list of array-like
    bits : int, default 12
        Resolution of the converter.
    peak_fraction : float, default 0.0035
        Peak of each channel as a fraction of the positive full scale.
    channels : sequence of int or None
        Channels that are quantized. ``None`` means all of them.

    Returns
    -------
    quantized : same container and shape as ``signals``
    """
    full_range = 20.0                      # volts, -10 V to +10 V; it cancels out of the result
    lsb = full_range / (2 ** int(bits))
    records, restore = as_records(signals)
    quantized = []
    for record in records:
        out = record.copy()
        for ch in _channel_list(channels, record.shape[1]):
            s = record[:, ch]
            peak = np.max(np.abs(s))
            if peak < 1e-14 or np.std(s) < 1e-14:      # channels that are constant inside a record stay untouched
                continue
            gain = (peak_fraction * full_range / 2) / peak
            out[:, ch] = np.round(s * gain / lsb) * lsb / gain
        quantized.append(out)
    return restore(quantized)


def _canonical(kind):
    name = str(kind).lower()
    name = _ALIASES.get(name, name)
    if name not in NOISE_KINDS:
        raise ValueError(f"Unknown noise kind {kind!r}; choose one of {NOISE_KINDS}.")
    return name


def _generator(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _channel_list(channels, n_channels):
    return range(n_channels) if channels is None else [int(c) for c in channels]
