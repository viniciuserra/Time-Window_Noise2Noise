"""Rebuild the result tables of the paper from the consolidated results.

    python paper/tables.py

Reads the values of the individual trials of the paper (``paper/reference/*.csv``)
and writes every result table of the paper to ``paper/TABLES.md``. The numbers
of the tables are the same as in the paper; ``build_tables`` returns them as
lists of rows for programmatic use.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from twn2n import benchmarks  # noqa: E402

REFERENCE = os.path.join(HERE, "reference")
WHITE_SNRS = (25.0, 20.0, 15.0, 12.0, 10.0, 5.0)
NOISES = (("white", "White"), ("pink", "Pink"), ("brown", "Brown"), ("quantization", "Quantization"))
CONFIGURATIONS = ("i", "ii", "iii", "iv", "v", "vi", "vii")


def read(name):
    return pd.read_csv(os.path.join(REFERENCE, name))


def gain(frame, method, band=None):
    suffix = "_snr" if band is None else f"_snr_{band}"
    return frame[method + suffix] - frame["noisy" + suffix]


def power_of_ten(p):
    if p >= 1e-3:
        return f"{p:.3f}"
    mantissa, exponent = f"{p:.1e}".split("e")
    return f"{mantissa} x 10^{int(exponent)}"


def build_tables():
    """Return ``{title: (header, rows)}`` with the result tables of the paper."""
    vib, damping, baseline = read("vibration_trials.csv"), read("damping_trials.csv"), read("baseline_trials.csv")
    impact, ablation = read("impact_trials.csv"), read("impact_no_metadata_trials.csv")
    white = {snr: vib[(vib.noise == "white") & (vib.snr_db == snr)] for snr in WHITE_SNRS}
    tables = {}

    rows = []
    for snr, s in white.items():
        g = gain(s, "twn2n")
        rows.append([f"{snr:.0f}", f"{s.twn2n_snr.mean():.2f}", f"{s.twn2n_snr.std():.2f}", f"{g.mean():.2f}",
                     f"{g.median():.2f}", f"{g.min():.2f}", f"{g.max():.2f}"])
    tables["Table 1. TWN2N under white noise (30 trials per level, dB)"] = (
        ["Input SNR", "Mean output", "Std dev", "Mean gain", "Median gain", "Min gain", "Max gain"], rows)

    rows = []
    for snr, s in white.items():
        rows.append([f"{snr:.0f}", f"{s.twn2n_snr.mean():.2f}", f"{gain(s, 'twn2n').mean():.2f}",
                     f"{s.lowpass_snr.mean():.2f}", f"{gain(s, 'lowpass').mean():.2f}",
                     f"{gain(s, 'twn2n', 'inband').mean():.2f}", f"{gain(s, 'lowpass', 'inband').mean():.2f}"])
    tables["Table 2. TWN2N against the low-pass filter with cutoff at 1.05 f3 (dB)"] = (
        ["Input SNR", "TWN2N output", "TWN2N gain", "LP output", "LP gain", "In-band gain TWN2N", "In-band gain LP"], rows)

    rows = []
    for snr, s in white.items():
        for method, label in (("twn2n", "TWN2N"), ("lowpass", "LP"), ("visushrink", "VisuShrink"), ("savgol", "SG(5,3)")):
            rows.append([f"{snr:.0f}", label] + [f"{gain(s, method, band).mean():+.2f}" for band in (None, "inband", "v1", "v2")])
    tables["Table 3. SNR gains resolved by band (dB)"] = (["Input SNR", "Method", "Full band", "0 to 1.25 f3", "V1", "V2"], rows)

    rows = []
    for noise, label in NOISES:
        s = vib[(vib.noise == noise) & (vib.snr_db == 15.0)]
        rows.append([label, f"{s.noisy_snr.mean():.2f}", f"{s.twn2n_snr.mean():.2f}", f"{s.twn2n_snr.std():.2f}",
                     f"{gain(s, 'twn2n').mean():.2f}", f"{gain(s, 'lowpass').mean():.2f}",
                     f"{gain(s, 'twn2n', 'inband').mean():.2f}", f"{gain(s, 'lowpass', 'inband').mean():.2f}"])
    tables["Table 4. TWN2N under different noise types (30 trials, dB)"] = (
        ["Noise type", "Input SNR", "Mean output", "Std dev", "Mean gain", "LP gain", "In-band gain TWN2N", "In-band gain LP"], rows)

    system = benchmarks.three_dof_system()
    tables["Table 5. Theoretical modal parameters"] = (
        ["Mode", "Natural frequency (rad/s)", "Natural frequency (Hz)", "Damping ratio"],
        [[str(k + 1), f"{system.wn[k]:.2f}", f"{system.fn[k]:.2f}", f"{system.zeta[k]:.4f}"] for k in range(3)])

    rows = []
    for snr in (20.0, 15.0):
        s = white[snr][white[snr].trial == 0].iloc[0]
        for k in (1, 2, 3):
            rows.append([f"{snr:.0f}", str(k), f"{s[f'noisy_itd_wn{k}']:.2f}", f"{s[f'noisy_itd_zeta{k}']:.3f}",
                         f"{s[f'twn2n_itd_wn{k}']:.2f}", f"{s[f'twn2n_itd_zeta{k}']:.3f}"])
    tables["Table 6. Modal parameters identified by the ITD method, one trial"] = (
        ["Input SNR", "Mode", "Noisy wn (rad/s)", "Noisy zeta", "TWN2N wn (rad/s)", "TWN2N zeta"], rows)

    rows = []
    for snr in (20.0, 15.0):
        s = white[snr]
        for k in (1, 2, 3):
            cells = [f"{snr:.0f}", str(k), f"{system.wn[k - 1]:.1f}", f"{system.zeta[k - 1]:.4f}"]
            for who in ("noisy", "twn2n"):
                found = s[f"{who}_itd_found{k}"].astype(bool)
                wn, zeta = s.loc[found, f"{who}_itd_wn{k}"], s.loc[found, f"{who}_itd_zeta{k}"]
                cells += [f"{wn.mean():.1f} ± {wn.std():.1f} ({int(found.sum())}/{len(s)})", f"{zeta.mean():.4f} ± {zeta.std():.4f}"]
            rows.append(cells)
    tables["Table 7. Modal parameters identified by the ITD method, 30 trials (mean ± standard deviation)"] = (
        ["Input SNR", "Mode", "Reference wn", "Reference zeta", "Noisy wn", "Noisy zeta", "TWN2N wn", "TWN2N zeta"], rows)

    rows = []
    for configuration in CONFIGURATIONS:
        s = damping[damping.configuration == configuration]
        first = s.iloc[0]
        cells = [f"({configuration})", f"{first.zeta_ref1:.4f} / {first.zeta_ref2:.4f} / {first.zeta_ref3:.4f}",
                 f"{gain(s, 'twn2n').mean():.2f} ± {gain(s, 'twn2n').std():.2f}", f"{gain(s, 'lowpass').mean():.2f}",
                 f"{gain(s, 'twn2n', 'inband').mean():.2f} / {gain(s, 'lowpass', 'inband').mean():.2f}"]
        for who in ("noisy", "twn2n"):
            for key, digits in (("wn_error_pct", 2), ("zeta_error_pct", 0)):
                medians = []
                for k in (1, 2, 3):
                    found = s[f"{who}_itd_found{k}"].astype(bool)
                    medians.append(f"{s.loc[found, f'{who}_itd_{key}{k}'].median():.{digits}f}")
                cells.append("/".join(medians))
        rows.append(cells)
    tables["Table 8. Sensitivity to the damping level at 15 dB input (30 trials per configuration)"] = (
        ["Configuration", "Damping ratios", "Gain TWN2N (dB)", "Gain LP (dB)", "In-band gain TWN2N / LP (dB)",
         "ITD noisy, wn error (%)", "ITD noisy, zeta error (%)", "ITD TWN2N, wn error (%)", "ITD TWN2N, zeta error (%)"], rows)

    methods = ("savgol", "savgol15", "visushrink_third", "visushrink", "visushrink_triple", "twn2n")
    header = ["SG(5,3)", "SG(15,3)", "VisuShrink sigma/3", "VisuShrink sigma", "VisuShrink 3 sigma", "TWN2N"]
    tables["Table 9. Output SNR of the classical filters, white noise (dB)"] = (
        ["Input SNR"] + header, [[f"{snr:.0f}"] + [f"{s[m + '_snr'].mean():.2f}" for m in methods] for snr, s in white.items()])
    rows = []
    for noise, label in NOISES:
        s = vib[(vib.noise == noise) & (vib.snr_db == 15.0)]
        rows.append([label, f"{s.noisy_snr.mean():.2f}"] + [f"{s[m + '_snr'].mean():.2f}" for m in methods])
    tables["Table 10. Output SNR of the classical filters, noise types (dB)"] = (["Noise type", "Input SNR"] + header, rows)

    rows, phys = [], []
    for pct in sorted(impact.noise_pct.unique()):
        s, a = impact[impact.noise_pct == pct].sort_values("trial"), ablation[ablation.noise_pct == pct]
        reduction = 100 * (1 - s.twn2n_distance / s.noisy_distance)
        every = np.logical_and.reduce([[flag == "1" for flag in str(f).zfill(20)] for f in s.improved_flags])
        t, p = ttest_rel(s.twn2n_distance, s.noisy_distance, alternative="less")
        rows.append([f"{s.snr_db.mean():.2f}", f"{s.noisy_distance.mean():.3f}", f"{s.twn2n_distance.mean():.3f}",
                     f"{reduction.min():.0f} to {reduction.max():.0f}", f"{int(every.sum())}/{len(every)}", f"{t:.1f}",
                     power_of_ten(p), f"{a.twn2n_distance.mean():.3f}"])
        phys.append([f"{s.snr_db.mean():.2f}"] + [f"{s['noisy_' + key].mean():.{d}f} / {s['twn2n_' + key].mean():.{d}f}"
                                                 for key, d in (("nrmse_x", 3), ("nrmse_v", 3), ("nrmse_f", 3),
                                                                ("nrmse_f_law", 3), ("violations_pct", 1))])
    tables["Table 11. Impact dataset: distance to the clean state (3 trials per level)"] = (
        ["Input SNR (dB)", "Noisy", "Denoised", "Reduction (%)", "Improved impacts", "t statistic", "p-value",
         "Denoised, no metadata"], rows)
    tables["Table 12. Impact dataset: error per variable, noisy / denoised"] = (
        ["Input SNR (dB)", "NRMSE indentation", "NRMSE velocity", "NRMSE force", "NRMSE force (law)", "Violations (%)"], phys)

    g = baseline.autoencoder_snr - baseline.noisy_snr
    tables["Autoencoder without temporal window, 20 dB input (30 trials, dB)"] = (
        ["Mean output", "Std dev", "Mean gain"],
        [[f"{baseline.autoencoder_snr.mean():.2f}", f"{baseline.autoencoder_snr.std():.2f}", f"{g.mean():.2f}"]])
    return tables


def main():
    lines = ["# Result tables of the paper\n",
             "This file is written by `paper/tables.py` from the values of the individual trials of the paper, stored in "
             "`paper/reference/`. LP is the zero-phase Butterworth low-pass filter of fourth order with the cutoff fixed at "
             "1.05 f3; the in-band gain refers to the band from 0 to 1.25 f3; V1 and V2 are the bands between the "
             "resonances.\n"]
    for title, (header, rows) in build_tables().items():
        lines += [f"## {title}\n", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(HERE, "TABLES.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
