"""Monte-Carlo robustness study of the ITD-identified modal parameters.

For each input SNR and each of N independent realizations: draw fresh noise, identify
modes from the noisy signal, denoise with TWN2N, identify modes again. Report the
ensemble mean, the standard uncertainty u and the expanded uncertainty U = 2u (k = 2).

Usage:
    python scripts/run_uncertainty_study.py --runs 10 --snrs 20 15
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twn2n.benchmarks import (impulse_response, add_white_noise,
                              system_matrices, theoretical_modal_parameters)
from twn2n.model import denoise
from twn2n.modal import identify_modes, match_to_reference

FS = 1024.0


def aggregate(series):
    v = pd.to_numeric(series, errors="coerce").dropna().values
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    u = v.std(ddof=1) if len(v) > 1 else np.nan
    return v.mean(), u, (2 * u if len(v) > 1 else np.nan), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--snrs", type=float, nargs="+", default=[20.0, 15.0])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    _, _, _, A, _ = system_matrices()
    wn_ref, zeta_ref = theoretical_modal_parameters(A)
    print("Reference wn [rad/s]:", np.round(wn_ref, 3))
    print("Reference zeta      :", np.round(zeta_ref, 3))

    t, S, yout = impulse_response(fs=FS)
    rows = []

    for snr in args.snrs:
        for run in range(1, args.runs + 1):
            rng = np.random.default_rng(int(1000 * snr) + run)
            S_noisy = add_white_noise(S, snr, rng=rng)

            try:
                wn_n, z_n = match_to_reference(
                    identify_modes(S_noisy[:, 0:3, 0], FS), wn_ref)
            except Exception:
                wn_n, z_n = [np.nan] * 3, [np.nan] * 3

            try:
                S_den, _ = denoise(S_noisy)
                wn_d, z_d = match_to_reference(
                    identify_modes(S_den[:, 0:3, 0], FS), wn_ref)
            except Exception as exc:
                print("  denoise/ITD failed:", exc)
                wn_d, z_d = [np.nan] * 3, [np.nan] * 3

            for m in range(3):
                rows.append(dict(snr=snr, run=run, mode=m + 1,
                                 wn_noisy=wn_n[m], zeta_noisy=z_n[m],
                                 wn_den=wn_d[m], zeta_den=z_d[m]))
            pd.DataFrame(rows).to_csv(
                os.path.join(args.outdir, "per_run_modal.csv"), index=False)
            print(f"[done] SNR={snr:g} dB  run {run}/{args.runs}")

    df = pd.DataFrame(rows)
    summary = []
    for snr in args.snrs:
        for mode in (1, 2, 3):
            sub = df[(df["snr"] == snr) & (df["mode"] == mode)]
            wn_n = aggregate(sub["wn_noisy"])
            wn_d = aggregate(sub["wn_den"])
            summary.append(dict(snr=snr, mode=mode, wn_ref=wn_ref[mode - 1],
                                wn_noisy_mean=wn_n[0], wn_noisy_U=wn_n[2], wn_noisy_det=wn_n[3],
                                wn_den_mean=wn_d[0], wn_den_U=wn_d[2], wn_den_det=wn_d[3]))
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(args.outdir, "uncertainty_summary.csv"), index=False)
    print("\n" + sdf.round(3).to_string())


if __name__ == "__main__":
    main()
