"""Time-Window Noise2Noise (TWN2N): self-supervised blind denoising of multichannel signals.

Typical use::

    import twn2n

    denoised = twn2n.denoise(noisy, p=2, r=4)

where ``noisy`` is an array with one row per time sample and one column per
channel. See :class:`twn2n.TWN2N` for the options.
"""
from . import benchmarks, filters, metrics, modal, noise
from .denoiser import TWN2N, default_r, denoise
from .metrics import band_snr_db, snr_db, snr_gain_db
from .noise import NOISE_KINDS, add_noise, colored_noise, quantize
from .windows import blind_spot_windows

__version__ = "1.0.0"

__all__ = [
    "TWN2N", "denoise", "default_r", "blind_spot_windows",
    "add_noise", "colored_noise", "quantize", "NOISE_KINDS",
    "snr_db", "snr_gain_db", "band_snr_db",
    "benchmarks", "filters", "metrics", "modal", "noise",
]
