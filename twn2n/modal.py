"""Modal identification: Ibrahim Time Domain (ITD) and Least-Squares Complex Exponential."""

import numpy as np


def ibrahim_time_domain(responses, fs, delay_block=20, model_order=10):
    """Generalized (oversized) ITD. Returns the continuous-time complex poles [rad/s]."""
    n_samples, n_dof = responses.shape
    dt_step = 1
    rows = model_order * n_dof
    L = n_samples - (model_order * delay_block) - dt_step
    if L <= rows:
        raise ValueError("Signal too short for the requested ITD order/delay.")

    Phi = np.zeros((rows, L))
    Phi_hat = np.zeros((rows, L))
    data = responses.T
    for i in range(model_order):
        s0 = i * delay_block
        r0, r1 = i * n_dof, (i + 1) * n_dof
        Phi[r0:r1, :] = data[:, s0:s0 + L]
        Phi_hat[r0:r1, :] = data[:, s0 + dt_step:s0 + L + dt_step]

    A_mat, *_ = np.linalg.lstsq(Phi.T, Phi_hat.T, rcond=None)
    z_vals, _ = np.linalg.eig(A_mat.T)
    return np.log(z_vals.astype(complex)) * fs


def poles_to_modes(poles, fs, min_sep=0.2, zeta_max=1.0):
    """Filter physical poles and return sorted, de-duplicated (wn [rad/s], zeta)."""
    modes = []
    for p in poles:
        if 1e-1 < p.imag < fs / 2.0:
            wn = abs(p)
            zeta = -p.real / wn
            if zeta < zeta_max:
                modes.append((wn, zeta))
    modes.sort(key=lambda m: m[0])

    unique, last = [], -1.0
    for wn, zeta in modes:
        if abs(wn - last) > min_sep:
            unique.append((wn, zeta))
            last = wn
    return unique


def identify_modes(responses, fs, delay_block=20):
    """Convenience wrapper: ITD + physical-pole filtering."""
    return poles_to_modes(ibrahim_time_domain(responses, fs, delay_block), fs)


def match_to_reference(modes, wn_ref, tol=10.0):
    """Match identified modes to reference frequencies in order.

    Returns (wn_est, zeta_est) with np.nan where a mode was not identified.
    """
    wn_est = [np.nan] * len(wn_ref)
    zeta_est = [np.nan] * len(wn_ref)
    for i, wr in enumerate(wn_ref):
        if i < len(modes) and abs(modes[i][0] - wr) < tol:
            wn_est[i], zeta_est[i] = modes[i]
    return wn_est, zeta_est


def least_squares_complex_exponential(y_signals, fs, order=6):
    """LSCEM (Prony-type) pole estimation from multiple output channels."""
    if y_signals.ndim == 1:
        y_signals = y_signals[:, np.newaxis]
    n_samples, n_channels = y_signals.shape
    L = n_samples - order - 1

    T_stack, y_stack = [], []
    for ch in range(n_channels):
        y = y_signals[:, ch]
        T = np.zeros((L, order))
        for i in range(L):
            T[i, :] = y[i:order + i][::-1]
        T_stack.append(T)
        y_stack.append(y[order:order + L])

    a, *_ = np.linalg.lstsq(np.vstack(T_stack), -np.concatenate(y_stack), rcond=None)
    roots = np.roots(np.concatenate(([1.0], a)))
    roots = roots[np.abs(roots) > 1e-12]
    return np.log(roots.astype(complex)) * fs
