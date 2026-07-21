"""Numerical benchmarks: linear 3-DOF mass-spring-damper system and noise models.

The 3-DOF system is simulated in state space; its unit impulse response provides
positions and accelerations with an exactly known ground truth, which is what makes a
quantitative evaluation of a blind denoiser possible.
"""

import numpy as np
from scipy import signal

# --- default physical parameters (as used in the paper) -----------------------
M_DEFAULT = (3.0, 3.0, 3.0)                 # masses [kg]
K_DEFAULT = (100.0, 125.0, 150.0, 200.0)    # stiffnesses [N/m]
C_DEFAULT = (5.0, 3.0, 2.0, 1.0)            # damping [Ns/m]
FS_DEFAULT = 1024.0                         # sampling frequency [Hz]
T_FINAL_DEFAULT = 10.0                      # simulation length [s]


def system_matrices(masses=M_DEFAULT, stiff=K_DEFAULT, damp=C_DEFAULT):
    """Return (M, K, C, A, B) for the 3-DOF chain system."""
    m1, m2, m3 = masses
    k1, k2, k3, k4 = stiff
    c1, c2, c3, c4 = damp

    M = np.diag([m1, m2, m3])
    K = np.array([[k1 + k2, -k2, 0.0],
                  [-k2, k2 + k3, -k3],
                  [0.0, -k3, k3 + k4]])
    C = np.array([[c1 + c2, -c2, 0.0],
                  [-c2, c2 + c3, -c3],
                  [0.0, -c3, c3 + c4]])

    n = 3
    Minv = np.linalg.inv(M)
    A_top = np.hstack([np.zeros((n, n)), np.identity(n)])
    A_bot = np.hstack([-Minv @ K, -Minv @ C])
    A = np.vstack([A_top, A_bot])
    B = np.vstack([np.zeros((n, n)), Minv])
    return M, K, C, A, B


def theoretical_modal_parameters(A):
    """Natural frequencies [rad/s] and damping ratios from the state matrix."""
    vals, _ = np.linalg.eig(A)
    poles = [s for s in vals if s.imag > 0 and s.real < 1e-6]
    poles.sort(key=lambda s: s.imag)
    wn, zeta, last = [], [], -1.0
    for s in poles:
        w = abs(s)
        if w > 0.1 and abs(w - last) > 0.1:
            wn.append(w)
            zeta.append(-s.real / w)
            last = w
    return np.array(wn), np.array(zeta)


def impulse_response(fs=FS_DEFAULT, t_final=T_FINAL_DEFAULT, **kwargs):
    """Simulate the unit impulse response of the 3-DOF system.

    Returns
    -------
    t : (n_samples,) time vector
    S : (n_samples, 6, 3) tensor -> [3 positions + 3 accelerations] x 3 excitations
    yout : (n_samples, 3, 3) positions only (clean ground truth)
    """
    _, _, _, A, B = system_matrices(**kwargs)
    n = 3
    A_bot = A[n:, :]
    C_sys = np.identity(2 * n)
    D = np.zeros((2 * n, n))
    t = np.arange(0.0, t_final, 1.0 / fs)

    positions, accelerations = [], []
    for i in range(n):
        sys_i = signal.StateSpace(A, B[:, i:i + 1], C_sys, D[:, i:i + 1])
        _, z = signal.impulse(sys_i, T=t)
        positions.append(z[:, :n])
        accelerations.append(z @ A_bot.T)

    yout = np.stack(positions, axis=2)
    aout = np.stack(accelerations, axis=2)
    S = np.concatenate((yout, aout), axis=1)
    return t, S, yout


def add_white_noise(data, snr_db, rng=None):
    """Add Gaussian white noise per channel at a target SNR in dB.

    SNR_dB = 20 * log10(std(signal) / std(noise)).
    """
    rng = np.random.default_rng() if rng is None else rng
    out = np.copy(data)
    for i in range(data.shape[2]):
        for j in range(data.shape[1]):
            sig = data[:, j, i]
            sd = np.std(sig)
            if sd < 1e-10:
                continue
            out[:, j, i] += rng.normal(0.0, sd / (10 ** (snr_db / 20.0)), data.shape[0])
    return out


def add_colored_noise(data, snr_db, kind="pink", rng=None):
    """Add pink (1/f) or brown (1/f^2) noise per channel at a target SNR in dB."""
    rng = np.random.default_rng() if rng is None else rng
    exponent = {"pink": 1.0, "brown": 2.0}[kind]
    out = np.copy(data)
    n = data.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / (freqs[1:] ** (exponent / 2.0))
    for i in range(data.shape[2]):
        for j in range(data.shape[1]):
            sig = data[:, j, i]
            sd = np.std(sig)
            if sd < 1e-10:
                continue
            spec = np.fft.rfft(rng.normal(size=n)) * scale
            noise = np.fft.irfft(spec, n=n)
            noise *= (sd / (10 ** (snr_db / 20.0))) / (np.std(noise) + 1e-30)
            out[:, j, i] += noise
    return out


def add_quantization_noise(data, n_bits=12, full_scale=20.0):
    """Quantize the signal with an ADC of `n_bits` over a symmetric full-scale range."""
    delta = full_scale / (2 ** n_bits)
    return delta * np.round(data / delta)


def snr_db(clean, estimate):
    """Global SNR in dB of `estimate` with respect to `clean`."""
    return 20.0 * np.log10(np.std(clean) / np.std(estimate - clean))
