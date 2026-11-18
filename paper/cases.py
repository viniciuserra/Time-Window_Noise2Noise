"""The two case studies of the paper, written with the public API of the library.

``vibration_case`` and ``impact_case`` run one trial each: they generate the
clean benchmark, corrupt it with the noise realization of the paper, train
TWN2N on the noisy signals, and return the metrics reported in the paper.

The random seeds follow the paper: trial ``k`` uses the noise realization
``k`` and the training seed ``1000 * k + 7``.
"""
import time

import numpy as np

import twn2n
from twn2n import benchmarks, filters, modal

P = 2                        # neighbors on each side of the masked instant
R_VIBRATION = 4         # bottleneck of the 3-DOF benchmark (six channels)
R_IMPACT = 3            # bottleneck of the impact benchmark (five channels)
LOWPASS_FACTOR = 1.05        # cutoff of the low-pass baseline, as a multiple of f3
INBAND_FACTOR = 1.25         # upper limit of the structural band, as a multiple of f3
DISPLACEMENTS = [0, 1, 2]    # channels x1, x2, x3 of the 3-DOF benchmark
ITD_DELAY, ITD_ORDER = 1, 20
ITD_WINDOW, ITD_ZETA_MAX = 0.05, 0.5

VIBRATION_SNRS = (25.0, 20.0, 15.0, 12.0, 10.0, 5.0)     # white-noise sweep, dB
NOISE_TYPES_SNR = 15.0                                   # level of the noise-type study, dB
IMPACT_NOISE_PCT = (3, 5, 7, 10, 15)                     # noise std as % of the signal std

_NOISE_ID = {"white": 1, "pink": 2, "brown": 3, "quantization": 4}


def paper_rng(noise, snr_db, trial, damping="R010"):
    """Random generator of the noise realization used in the paper."""
    return np.random.default_rng([20260930, int(round(snr_db * 10)), int(trial), _NOISE_ID[noise],
                                  sum(ord(ch) for ch in damping)])


def training_seed(trial):
    return 1000 * int(trial) + 7


# ----------------------------------------------------------- vibration case --
def vibration_signals(noise="white", snr_db=15.0, trial=0, damping="R010"):
    """Clean and noisy impulse responses of the 3-DOF benchmark for one trial.

    Returns ``(system, t, clean, noisy)``; ``clean`` and ``noisy`` are lists of
    three records with shape ``(n_samples, 6)``.
    """
    system = benchmarks.three_dof_system(damping)
    t, clean = benchmarks.three_dof_impulse_responses(system)
    noisy = twn2n.add_noise(clean, noise, snr_db, paper_rng(noise, snr_db, trial, damping))
    return system, t, clean, noisy


def vibration_case(noise="white", snr_db=15.0, trial=0, damping="R010", epochs=5000, verbose=0,
                   return_signals=False):
    """One trial of the 3-DOF vibration study.

    Parameters
    ----------
    noise : {"white", "pink", "brown", "quantization"}
    snr_db : float
        Input SNR of every channel (ignored for quantization).
    trial : int
        Index of the trial (0 to 29 in the paper).
    damping : str
        Damping configuration (see ``twn2n.benchmarks.THREE_DOF_DAMPING``).
    epochs : int
        Maximum number of epochs (5,000 in the paper).
    return_signals : bool
        If ``True``, also return a dictionary with the signals.

    Returns
    -------
    result : dict
        SNR of the displacements, in dB, for the noisy input, TWN2N, and the
        classical filters; SNR inside the structural band; modal parameters
        identified by the ITD method; and training information.
    """
    start = time.time()
    system, t, clean, noisy = vibration_signals(noise, snr_db, trial, damping)
    f3 = system.fn[2]

    model = twn2n.TWN2N(p=P, r=R_VIBRATION, seed=training_seed(trial), epochs=epochs,
                        verbose=verbose)
    denoised = model.fit_denoise(noisy)

    noise_std = float(np.std(np.vstack(noisy)[:, DISPLACEMENTS] - np.vstack(clean)[:, DISPLACEMENTS]))
    estimates = {
        "noisy": noisy,
        "twn2n": denoised,
        "lowpass": filters.lowpass(noisy, LOWPASS_FACTOR * f3, system.fs),
        "savgol": filters.savitzky_golay(noisy, window=5, polyorder=3),
        "visushrink": filters.visushrink(noisy, sigma=noise_std),
    }
    band = (0.0, INBAND_FACTOR * f3)

    result = dict(case="vibration", noise=noise, snr_db=float(snr_db), trial=int(trial), damping=damping,
                  p=P, r=R_VIBRATION, fs=system.fs, duration=system.duration,
                  fn_hz=[float(v) for v in system.fn], zeta=[float(v) for v in system.zeta],
                  lowpass_cutoff_hz=float(LOWPASS_FACTOR * f3), n_rows=int(sum(len(r) for r in noisy)))
    for name, estimate in estimates.items():
        result[f"{name}_snr"] = twn2n.snr_db(clean, estimate, channels=DISPLACEMENTS)
        result[f"{name}_snr_inband"] = twn2n.band_snr_db(clean, estimate, band, system.fs, channels=DISPLACEMENTS)
    for name in ("clean", "noisy", "twn2n", "lowpass"):
        signals = clean if name == "clean" else estimates[name]
        result.update(_itd(signals[0][:, DISPLACEMENTS], system, name))
    result.update(epochs_run=model.epochs_run_, best_epoch=model.best_epoch_,
                  training_time_s=model.training_time_s_, n_parameters=model.n_parameters_,
                  wall_time_s=time.time() - start)
    if return_signals:
        return result, dict(t=t, clean=clean, system=system, model=model, **estimates)
    return result


def _itd(responses, system, prefix):
    """ITD identification on the displacements of the first record (impulse at mass 1)."""
    try:
        modes = modal.identify_modes(responses, system.fs, delay=ITD_DELAY, order=ITD_ORDER)
        matches = modal.match_modes(modes, system.wn, system.zeta, ITD_WINDOW, ITD_ZETA_MAX)
    except (ValueError, np.linalg.LinAlgError):
        matches = [dict(mode=k + 1, found=False, wn=np.nan, zeta=np.nan) for k in range(3)]
    out = {}
    for match in matches:
        k = match["mode"]
        out[f"{prefix}_itd_wn{k}"] = float(match["wn"])
        out[f"{prefix}_itd_zeta{k}"] = float(match["zeta"])
        out[f"{prefix}_itd_found{k}"] = bool(match["found"])
    return out


# -------------------------------------------------------------- impact case --
def impact_signals(noise_pct=10, trial=0):
    """Clean and noisy signals of the impact benchmark for one trial.

    Gaussian noise with a standard deviation of ``noise_pct`` % of the standard
    deviation of each signal is added to the indentation, the velocity, and the
    force of each impact. The two test conditions (restitution and approach
    velocity) are known and stay noise-free.

    Returns ``(clean, noisy)``, two lists of twenty records ``(n_samples, 5)``.
    """
    clean, _ = benchmarks.hunt_crossley_impacts()
    rng = np.random.default_rng([20260930, int(noise_pct), int(trial), 77])
    snr_db = 20 * np.log10(100.0 / noise_pct)
    noisy = twn2n.add_noise(clean, "white", snr_db, rng, channels=[0, 1, 2], exact_snr=False)
    return clean, noisy


def impact_case(noise_pct=10, trial=0, epochs=5000, verbose=0, return_signals=False):
    """One trial of the nonlinear impact study.

    Returns
    -------
    result : dict
        Dimensionless distance to the clean state (each variable divided by
        the standard deviation of its clean signal), normalized root-mean-square
        error (NRMSE) per variable, NRMSE of the force recomputed from the
        kinematic signals with the Hunt and Crossley law, and number of impacts
        whose distance decreased. Values are averages over the twenty impacts,
        on the samples of the contact interval.
    """
    start = time.time()
    clean, noisy = impact_signals(noise_pct, trial)
    model = twn2n.TWN2N(p=P, r=R_IMPACT, network="compact", seed=training_seed(trial),
                        epochs=epochs, verbose=verbose)
    denoised = model.fit_denoise(noisy)

    result = dict(case="impact", noise_pct=float(noise_pct), snr_db=float(20 * np.log10(100.0 / noise_pct)),
                  trial=int(trial), p=P, r=R_IMPACT, n_impacts=len(clean),
                  n_rows=int(sum(len(r) for r in clean)))
    scale = np.vstack(clean)[:, :3].std(axis=0)
    per_impact = {"noisy": [], "twn2n": []}
    for name, estimate in (("noisy", noisy), ("twn2n", denoised)):
        for c, e in zip(clean, estimate):
            contact = c[:, 0] > 0
            error = (e[contact, :3] - c[contact, :3]) / scale
            law = benchmarks.hunt_crossley_force(e[contact, 0], e[contact, 1], c[contact, 3], c[contact, 4]) / 1000.0
            per_impact[name].append(dict(
                distance=np.sqrt((error ** 2).mean()),
                nrmse_x=np.sqrt((error[:, 0] ** 2).mean()), nrmse_v=np.sqrt((error[:, 1] ** 2).mean()),
                nrmse_f=np.sqrt((error[:, 2] ** 2).mean()),
                nrmse_f_law=np.sqrt(((law - c[contact, 2]) ** 2).mean()) / scale[2]))
        for key in ("distance", "nrmse_x", "nrmse_v", "nrmse_f", "nrmse_f_law"):
            result[f"{name}_{key}"] = float(np.mean([row[key] for row in per_impact[name]]))
    d_noisy = np.array([row["distance"] for row in per_impact["noisy"]])
    d_twn2n = np.array([row["distance"] for row in per_impact["twn2n"]])
    result["impacts_improved"] = int((d_twn2n < d_noisy).sum())
    result["distance_per_impact_noisy"] = [float(v) for v in d_noisy]
    result["distance_per_impact_twn2n"] = [float(v) for v in d_twn2n]
    for k, name in enumerate(("x", "v", "f")):
        result[f"noisy_snr_{name}"] = twn2n.snr_db(clean, noisy, channels=[k])
        result[f"twn2n_snr_{name}"] = twn2n.snr_db(clean, denoised, channels=[k])
    result.update(epochs_run=model.epochs_run_, best_epoch=model.best_epoch_,
                  training_time_s=model.training_time_s_, n_parameters=model.n_parameters_,
                  wall_time_s=time.time() - start)
    if return_signals:
        return result, dict(clean=clean, noisy=noisy, twn2n=denoised, model=model)
    return result
