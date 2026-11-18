# Case studies of the paper

This folder contains the consolidated results of the paper, the script that rebuilds its tables from them, the two
case studies written with the library, and example trials run with the library.

| File or folder | Content |
|---|---|
| `cases.py` | One trial of each case study, written with the public functions of the library: `vibration_case` (3-DOF system) and `impact_case` (contact law of Hunt and Crossley). |
| `run.py` | Command-line runner. It runs single trials or sets of trials, in parallel, and stores one JSON file per trial in `results/trials/`. |
| `summarize.py` | Gathers the trials into `results/*.csv` and writes the comparison tables in `RESULTS.md`. |
| `reference/` | Consolidated results of the paper, one row per trial: the vibration study (`vibration_trials.csv`), the damping configurations (`damping_trials.csv`), the autoencoder without temporal window (`baseline_trials.csv`), and the impact study with and without the test conditions as inputs (`impact_trials.csv`, `impact_no_metadata_trials.csv`). |
| `tables.py` | Rebuilds every result table of the paper from `reference/` and writes `TABLES.md`. |
| `TABLES.md` | The result tables of the paper. |
| `results/` | Example trials run with this library: one trial of each condition. |
| `RESULTS.md` | The example trials next to the values of the paper. |

## Repetitions in the paper

A single trial depends on the noise realization and on the initial weights of the network, so the paper repeats
every condition and reports statistics over the trials:

- **Vibration of the 3-DOF system:** white noise at six input SNRs (25, 20, 15, 12, 10, and 5 dB) and pink noise,
  brown noise, and quantization at about 15 dB. Each of the nine conditions has thirty trials.
- **Nonlinear impact:** five noise levels, with the standard deviation of the noise at 3, 5, 7, 10, and 15% of
  that of each signal. Each level has three trials, and every trial contains the twenty impacts of the dataset.

Trial `k` uses the noise realization `k` and the training seed `1000 k + 7`. The examples stored in `results/` are
trial 0 of each condition.

## Settings

Both case studies use the settings of the paper: two neighbors on each side of the masked instant (`p = 2`); a
bottleneck of four units for the six channels of the vibration case and of three units for the five channels of the
impact case; and the default training protocol of the library.

The low-pass baseline is a zero-phase Butterworth filter of fourth order with the cutoff fixed at 1.05 times the
third natural frequency of the structure.

## Commands

Run from the root of the repository:

```bash
python paper/run.py vibration --noise white --snr 15 --trials 0          # one trial
python paper/run.py vibration --study all --trials 0                     # one trial of each of the nine conditions
python paper/run.py vibration --study all --trials 0-29 --workers 6      # the thirty trials of each condition
python paper/run.py impact --study all --trials 0 --workers 5            # one trial of each noise level
python paper/summarize.py                                                # example trials next to the paper, in RESULTS.md
python paper/tables.py                                                   # result tables of the paper, in TABLES.md
```

Each worker is a separate process that uses two CPU threads and about 1 GB of memory. One trial of the vibration
case takes a few minutes, and one trial of the impact case takes about one hour. Trials that already
have a result file are skipped, so an interrupted run resumes with the same command.

## What to expect from a new trial

The signals, the noise realizations, the classical filters, the metrics, and the modal identification are
deterministic: the library gives the values of the paper for them. The output of TWN2N depends on the initial
weights of the network, which change with the software version, so a new trial is expected to fall inside the
range of the trials of the paper and not to repeat one of them. In the impact case the paper has three trials per
noise level, so its range is narrow, and a new trial can fall slightly outside it.
