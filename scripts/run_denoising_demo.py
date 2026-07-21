"""Single-run demo: denoise the 3-DOF benchmark and report the SNR gain.

Usage:
    python scripts/run_denoising_demo.py --snr 20
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twn2n.benchmarks import (impulse_response, add_white_noise, snr_db,
                              system_matrices, theoretical_modal_parameters)
from twn2n.model import denoise
from twn2n.modal import identify_modes, match_to_reference

FS = 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=20.0, help="input SNR in dB")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _, _, _, A, _ = system_matrices()
    wn_ref, _ = theoretical_modal_parameters(A)

    t, S, yout = impulse_response(fs=FS)
    rng = np.random.default_rng(args.seed)
    S_noisy = add_white_noise(S, args.snr, rng=rng)

    print(f"Input SNR (target): {args.snr:.2f} dB")
    print("Training TWN2N on the noisy signal only ...")
    S_den, _ = denoise(S_noisy)

    out = snr_db(yout, S_den[:, 0:3, :])
    print(f"Output SNR        : {out:.2f} dB")
    print(f"SNR gain          : {out - args.snr:+.2f} dB")

    print("\nModal identification (ITD), natural frequencies [rad/s]:")
    print("  reference:", np.round(wn_ref, 2))
    for label, sig in (("noisy   ", S_noisy[:, 0:3, 0]), ("denoised", S_den[:, 0:3, 0])):
        try:
            wn, _ = match_to_reference(identify_modes(sig, FS), wn_ref)
            print(f"  {label}:", [("--" if np.isnan(w) else round(w, 2)) for w in wn])
        except Exception as exc:
            print(f"  {label}: identification failed ({exc})")


if __name__ == "__main__":
    main()
