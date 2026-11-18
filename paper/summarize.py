"""Gather the trials run with ``paper/run.py`` and compare them with the paper.

    python paper/summarize.py

Reads ``paper/results/trials/*.json`` and the values reported in the paper
(``paper/reference/*.csv``). Writes ``paper/results/vibration_trials.csv``,
``paper/results/impact_trials.csv``, and the comparison tables in
``paper/RESULTS.md``.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRIALS = os.path.join(HERE, "results", "trials")
RESULTS = os.path.join(HERE, "results")
REFERENCE = os.path.join(HERE, "reference")
CONDITIONS = ([("white", snr) for snr in (25.0, 20.0, 15.0, 12.0, 10.0, 5.0)]
              + [(noise, 15.0) for noise in ("pink", "brown", "quantization")])


def load_trials(case):
    """Example trials of one case study.

    Starts from the trials stored in ``results/<case>_trials.csv`` and adds or
    replaces the trials found as JSON files in ``results/trials`` (the output
    of ``paper/run.py``). With no new trial, the stored ones are returned.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(TRIALS, f"{case}_*.json"))):
        with open(path, encoding="utf-8") as handle:
            row = json.load(handle)
        environment = row.pop("environment", {})
        row.update({f"env_{k}": v for k, v in environment.items()})
        rows.append({k: v for k, v in row.items() if not isinstance(v, (list, dict))})
    new = pd.DataFrame(rows)
    stored_path = os.path.join(RESULTS, f"{case}_trials.csv")
    stored = pd.read_csv(stored_path) if os.path.exists(stored_path) else pd.DataFrame()
    if new.empty or stored.empty:
        return stored if new.empty else new
    keys = ["noise", "snr_db", "trial", "damping"] if case == "vibration" else ["noise_pct", "trial"]
    return pd.concat([stored, new], ignore_index=True).drop_duplicates(subset=keys, keep="last")


def value(values, digits=2):
    """One value as it is; several values as mean ± standard deviation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "not found"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def spread(values, digits=2):
    """Mean ± standard deviation and range of the trials of the paper."""
    values = np.asarray(values, dtype=float)
    return f"{value(values, digits)} ({values.min():.{digits}f} to {values.max():.{digits}f})"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def trial_label(frame):
    trials = sorted(frame.trial.unique())
    return f"trial {trials[0]}" if len(trials) == 1 else f"{len(trials)} trials"


# ---------------------------------------------------------------- vibration --
def vibration_tables(lib, ref):
    out = []
    label = trial_label(lib)

    rows = []
    for noise, snr in CONDITIONS:
        a = ref[(ref.noise == noise) & (ref.snr_db == snr)]
        b = lib[(lib.noise == noise) & (lib.snr_db == snr)]
        if b.empty:
            continue
        rows.append([noise, f"{a.noisy_snr.mean():.2f}", spread(a.twn2n_snr), value(b.twn2n_snr),
                     f"{(a.twn2n_snr - a.noisy_snr).mean():+.2f}", f"{(b.twn2n_snr - b.noisy_snr).mean():+.2f}"])
    out.append("### TWN2N output SNR\n\n"
               "SNR of the displacements after denoising, in dB. The paper column gives the mean ± standard "
               f"deviation of its thirty trials, with the range in parentheses. The library column gives {label}. "
               "The output of TWN2N depends on the initial weights of the network, so a trial of the library is "
               "expected to fall inside the range of the paper, not to repeat one of its trials.\n\n"
               + table(["Noise", "Input SNR", "Output, paper", "Output, library", "Mean gain, paper",
                        "Gain, library"], rows))

    rows = []
    for noise, snr in CONDITIONS:
        b = lib[(lib.noise == noise) & (lib.snr_db == snr)]
        a = ref[(ref.noise == noise) & (ref.snr_db == snr) & ref.trial.isin(b.trial)]
        if b.empty:
            continue
        cells = [noise, f"{a.noisy_snr.mean():.2f}"]
        for name in ("lowpass", "savgol", "visushrink"):
            cells.append(f"{(a[f'{name}_snr'] - a.noisy_snr).mean():+.2f}")
            cells.append(f"{(b[f'{name}_snr'] - b.noisy_snr).mean():+.2f}")
        rows.append(cells)
    out.append("### Classical filters\n\n"
               "SNR gain of the displacements, in dB, for the same trials in the paper and in the library: "
               "zero-phase Butterworth low-pass filter of fourth order with the cutoff fixed at 1.05 f3; "
               "Savitzky-Golay filter (window of five samples, third order); and VisuShrink with the exact noise "
               "level. These filters involve no training, so the two columns of each filter must coincide.\n\n"
               + table(["Noise", "Input SNR", "Low-pass, paper", "Low-pass, library", "Savitzky-Golay, paper",
                        "Savitzky-Golay, library", "VisuShrink, paper", "VisuShrink, library"], rows))

    rows = []
    for noise, snr in CONDITIONS:
        a = ref[(ref.noise == noise) & (ref.snr_db == snr)]
        b = lib[(lib.noise == noise) & (lib.snr_db == snr)]
        if b.empty:
            continue
        same = a[a.trial.isin(b.trial)]
        rows.append([noise, f"{a.noisy_snr.mean():.2f}",
                     spread(a.twn2n_snr_inband - a.noisy_snr_inband), value(b.twn2n_snr_inband - b.noisy_snr_inband),
                     f"{(same.lowpass_snr_inband - same.noisy_snr_inband).mean():+.2f}",
                     f"{(b.lowpass_snr_inband - b.noisy_snr_inband).mean():+.2f}"])
    out.append("### Gain inside the structural band\n\n"
               "SNR gain of the displacements restricted to the band from 0 to 1.25 f3, in dB. The TWN2N column "
               "of the paper gives the mean ± standard deviation and the range of its thirty trials; the "
               "low-pass columns compare the same trials.\n\n"
               + table(["Noise", "Input SNR", "TWN2N, paper", "TWN2N, library", "Low-pass, paper",
                        "Low-pass, library"], rows))

    rows = []
    for snr in (20.0, 15.0):
        a = ref[(ref.noise == "white") & (ref.snr_db == snr)]
        b = lib[(lib.noise == "white") & (lib.snr_db == snr)]
        if b.empty:
            continue
        same = a[a.trial.isin(b.trial)]
        for k in (1, 2, 3):
            found = a[f"twn2n_itd_found{k}"].astype(bool)
            rows.append([f"{snr:.0f}", k,
                         value(same.loc[same[f"noisy_itd_found{k}"].astype(bool), f"noisy_itd_zeta{k}"], 4),
                         value(b.loc[b[f"noisy_itd_found{k}"].astype(bool), f"noisy_itd_zeta{k}"], 4),
                         value(a.loc[found, f"twn2n_itd_zeta{k}"], 4),
                         value(b.loc[b[f"twn2n_itd_found{k}"].astype(bool), f"twn2n_itd_zeta{k}"], 4)])
    out.append("### Damping ratios identified by the ITD method\n\n"
               "Damping ratio of each mode identified from the displacements of the first record. The reference "
               "values are 0.0100, 0.0072, and 0.0100. The noisy columns compare the same trials and must "
               "coincide, because the identification from the noisy signals involves no training. The TWN2N "
               "column of the paper gives the mean ± standard deviation of its thirty trials.\n\n"
               + table(["Input SNR", "Mode", "Noisy, paper", "Noisy, library", "TWN2N, paper", "TWN2N, library"], rows))
    return out


# ------------------------------------------------------------------- impact --
def impact_tables(lib, ref):
    label = trial_label(lib)
    rows = []
    for pct in sorted(ref.noise_pct.unique()):
        a, b = ref[ref.noise_pct == pct], lib[lib.noise_pct == pct]
        if b.empty:
            continue
        rows.append([f"{a.snr_db.mean():.2f}", f"{a.noisy_distance.mean():.3f}", f"{b.noisy_distance.mean():.3f}",
                     spread(a.twn2n_distance, 4), value(b.twn2n_distance, 4)])
    first = ("### Distance to the clean state\n\n"
             "Dimensionless distance between the estimated and the clean state (indentation, velocity, and force, "
             "each divided by the standard deviation of its clean signal), averaged over the twenty impacts. The "
             "TWN2N column of the paper gives the mean ± standard deviation of its three trials, with the range in "
             f"parentheses; the library columns give {label}.\n\n"
             + table(["Input SNR", "Noisy, paper", "Noisy, library", "TWN2N, paper", "TWN2N, library"], rows))
    rows = []
    for pct in sorted(ref.noise_pct.unique()):
        a, b = ref[ref.noise_pct == pct], lib[lib.noise_pct == pct]
        if b.empty:
            continue
        cells = [f"{a.snr_db.mean():.2f}"]
        for key in ("nrmse_x", "nrmse_v", "nrmse_f", "nrmse_f_law"):
            cells.append(f"{a[f'twn2n_{key}'].mean():.3f}")
            cells.append(f"{b[f'twn2n_{key}'].mean():.3f}")
        rows.append(cells)
    second = ("### Error per variable after denoising\n\n"
              "Normalized root-mean-square error of the TWN2N output: indentation, velocity, measured force, and "
              "force recomputed from the kinematic signals with the Hunt and Crossley law. The paper columns give "
              "the mean of its three trials.\n\n"
              + table(["Input SNR", "x, paper", "x, library", "v, paper", "v, library", "F, paper", "F, library",
                       "F (law), paper", "F (law), library"], rows))
    return [first, second]


def environment_line(frame):
    columns = [c for c in ("env_python", "env_tensorflow", "env_numpy", "env_platform") if c in frame]
    if frame.empty or not columns:
        return "not recorded"
    combos = frame[columns].drop_duplicates().to_dict("records")
    return "; ".join("Python {env_python}, TensorFlow {env_tensorflow}, NumPy {env_numpy}, {env_platform}".format(**c)
                     for c in combos)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    sections = ["# Case studies of the paper, run with the library\n",
                "This file is written by `paper/summarize.py`. It compares the trials run with this library "
                "(`paper/run.py`, stored in `paper/results/`) with the values reported in the paper "
                "(`paper/reference/`). Every trial uses the noise realization of the corresponding trial of the "
                "paper. The result tables of the paper themselves are in `paper/TABLES.md`.\n"]

    vib = load_trials("vibration")
    if not vib.empty:
        vib = vib.sort_values(["noise", "snr_db", "trial"]).reset_index(drop=True)
        vib.to_csv(os.path.join(RESULTS, "vibration_trials.csv"), index=False, float_format="%.6f")
        ref = pd.read_csv(os.path.join(REFERENCE, "vibration_trials.csv"))
        sections.append("## Vibration of the 3-DOF system\n")
        sections.append(f"{len(vib)} trials run with the library ({environment_line(vib)}). Training took from "
                        f"{vib.training_time_s.min() / 60:.1f} to {vib.training_time_s.max() / 60:.1f} min per trial "
                        f"({int(vib.epochs_run.min())} to {int(vib.epochs_run.max())} epochs).\n")
        sections += vibration_tables(vib, ref)

    imp = load_trials("impact")
    if not imp.empty:
        imp = imp.sort_values(["noise_pct", "trial"]).reset_index(drop=True)
        imp.to_csv(os.path.join(RESULTS, "impact_trials.csv"), index=False, float_format="%.6f")
        ref = pd.read_csv(os.path.join(REFERENCE, "impact_trials.csv"))
        sections.append("\n## Nonlinear impact (Hunt and Crossley)\n")
        sections.append(f"{len(imp)} trials run with the library ({environment_line(imp)}). Training took from "
                        f"{imp.training_time_s.min() / 60:.0f} to {imp.training_time_s.max() / 60:.0f} min per "
                        "trial.\n")
        sections += impact_tables(imp, ref)

    text = "\n\n".join(s.strip("\n") for s in sections) + "\n"
    with open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
