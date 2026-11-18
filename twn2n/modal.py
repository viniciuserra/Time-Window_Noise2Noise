"""Modal identification with the Ibrahim Time Domain (ITD) method.

The paper uses the ITD method to show the effect of the denoising on a
quantity derived from the signals: the natural frequencies and the damping
ratios identified from free responses.
"""
import numpy as np


def ibrahim_time_domain(responses, fs, delay=1, order=20):
    """Poles and mode shapes of a free response, by the oversized ITD method.

    Parameters
    ----------
    responses : array-like, shape (n_samples, n_dof)
        Free responses (for example, the displacements after an impulse).
    fs : float
        Sampling frequency, in Hz.
    delay : int, default 1
        Delay, in samples, between the stacked blocks of the data matrix.
    order : int, default 20
        Number of stacked blocks. The model has ``order * n_dof`` poles, far
        more than the physical ones, which leaves room for the noise.

    Returns
    -------
    poles : numpy.ndarray of complex
        Continuous-time poles, in rad/s.
    mode_shapes : numpy.ndarray, shape (n_poles, n_dof)
    """
    responses = np.asarray(responses, dtype=float)
    n_samples, n_dof = responses.shape
    rows = order * n_dof
    length = n_samples - order * delay - 1
    if length <= rows:
        raise ValueError("The record is too short for the ITD method with this order and delay.")
    phi = np.zeros((rows, length))
    phi_shifted = np.zeros((rows, length))
    data = responses.T
    for i in range(order):
        start = i * delay
        phi[i * n_dof:(i + 1) * n_dof, :] = data[:, start:start + length]
        phi_shifted[i * n_dof:(i + 1) * n_dof, :] = data[:, start + 1:start + length + 1]
    system, _, _, _ = np.linalg.lstsq(phi.T, phi_shifted.T, rcond=None)
    eigenvalues, eigenvectors = np.linalg.eig(system.T)
    poles = np.log(eigenvalues.astype(complex)) * fs
    return poles, eigenvectors[:n_dof, :].T


def identify_modes(responses, fs, delay=1, order=20, tolerance=0.2):
    """Identify the oscillatory modes of a free response with the ITD method.

    Parameters
    ----------
    responses : array-like, shape (n_samples, n_dof)
    fs : float
    delay, order : int
        See :func:`ibrahim_time_domain`.
    tolerance : float, default 0.2
        Poles whose natural frequencies differ by less than this value, in
        rad/s, are merged.

    Returns
    -------
    modes : list of dict
        One entry per identified mode, sorted by natural frequency, with the
        keys ``"wn"`` (rad/s), ``"fn"`` (Hz), ``"zeta"``, ``"pole"``, and
        ``"mode_shape"``. The list contains spurious (computational) modes in
        addition to the physical ones; :func:`match_modes` selects the latter.
    """
    poles, shapes = ibrahim_time_domain(responses, fs, delay, order)
    modes = []
    for pole, shape in zip(poles, shapes):
        if 1e-1 < pole.imag < np.pi * fs:
            wn = np.abs(pole)
            zeta = -pole.real / wn
            if zeta < 1.0:
                modes.append(dict(pole=pole, wn=float(wn), fn=float(wn / (2 * np.pi)),
                                  zeta=float(zeta), mode_shape=shape))
    modes.sort(key=lambda m: m["wn"])
    unique, last = [], -1.0
    for mode in modes:
        if abs(mode["wn"] - last) > tolerance:
            unique.append(mode)
            last = mode["wn"]
    return unique


def match_modes(modes, wn_reference, zeta_reference=None, relative_window=0.05, zeta_max=0.5):
    """Match identified modes to reference natural frequencies.

    For each reference frequency, the nearest identified mode inside a relative
    window is taken. Candidates with a damping ratio of ``zeta_max`` or more are
    discarded as spurious.

    Parameters
    ----------
    modes : list of dict
        Output of :func:`identify_modes`.
    wn_reference : sequence of float
        Reference natural frequencies, in rad/s.
    zeta_reference : sequence of float or None
        Reference damping ratios, used only to report the errors.
    relative_window : float, default 0.05
    zeta_max : float, default 0.5

    Returns
    -------
    matches : list of dict
        One entry per reference mode with the keys ``"mode"``, ``"found"``,
        ``"wn"``, ``"fn"``, ``"zeta"``, ``"wn_error_pct"``, and
        ``"zeta_error_pct"`` (``nan`` where the mode was not found).
    """
    if zeta_reference is None:
        zeta_reference = [np.nan] * len(wn_reference)
    matches, used = [], set()
    for i, (w0, z0) in enumerate(zip(wn_reference, zeta_reference)):
        best, best_distance = None, np.inf
        for j, mode in enumerate(modes):
            if j in used or mode["zeta"] >= zeta_max:
                continue
            distance = abs(mode["wn"] - w0)
            if distance < relative_window * w0 and distance < best_distance:
                best, best_distance = j, distance
        if best is None:
            matches.append(dict(mode=i + 1, found=False, wn=np.nan, fn=np.nan, zeta=np.nan,
                                wn_error_pct=np.nan, zeta_error_pct=np.nan))
            continue
        used.add(best)
        mode = modes[best]
        matches.append(dict(mode=i + 1, found=True, wn=mode["wn"], fn=mode["fn"], zeta=mode["zeta"],
                            wn_error_pct=float(100 * abs(mode["wn"] - w0) / w0),
                            zeta_error_pct=float(100 * abs(mode["zeta"] - z0) / z0)))
    return matches
