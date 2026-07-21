# Time-Window Noise2Noise (TWN2N)

Reference implementation of **Time-Window Noise2Noise (TWN2N)** — a self-supervised
method for **blind denoising** of signals produced by deterministic dynamical
systems (vibration and impact signals in mechanical systems).

The method needs **no clean reference signal**, **no paired noisy observations**, and
**no prior characterization of the noise**. Each time instant is reconstructed from its
temporal neighbourhood while the instant itself is *masked* (a "blind spot" in time),
which forces a compact autoencoder to learn the underlying dynamics instead of the noise.

This repository accompanies the manuscript:

> V. S. Vianna, T. H. Machado, I. F. Santos,
> *Time-Window Noise2Noise: A Self-Supervised Method for Blind Denoising of Vibration
> and Impact Signals in Mechanical Systems*.

---

## Idea in one picture

For a window size `p`, the network input at instant `t_i` is

```
X_{i,p} = [ X(t_{i-p}), ..., X(t_{i-1}),  X(t_{i+1}), ..., X(t_{i+p}) ]
                                  ^^^ X(t_i) is deliberately excluded
```

and the training target is the **noisy** sample `X(t_i)` itself. Because the central
instant is never shown to the network, it cannot learn the identity map; combined with a
bottleneck smaller than the output dimension, the only way to predict `X(t_i)` is to learn
the deterministic dynamics. The uncorrelated noise cannot be predicted and is suppressed.

---

## Installation

```bash
git clone https://github.com/viniciuserra/Time-Window_Noise2Noise.git
cd Time-Window_Noise2Noise
pip install -r requirements.txt
```

Python 3.9+ is recommended. A GPU is **not** required — the models are small
(~17k parameters) and train in a few minutes on a standard CPU.

---

## Quick start

Denoise the 3-DOF benchmark at 20 dB SNR and report the SNR gain:

```bash
python scripts/run_denoising_demo.py --snr 20
```

Reproduce the Monte-Carlo robustness / uncertainty study of the identified modal
parameters (this is what produces the uncertainty table in the paper):

```bash
python scripts/run_uncertainty_study.py --runs 10 --snrs 20 15
```

Results are written to `results/`.

---

## Repository layout

```
twn2n/
  benchmarks.py   3-DOF mass-spring-damper system, impulse response, noise models
  model.py        TWN2N architecture, time-window (blind-spot) dataset, training, inference
  modal.py        Modal identification: Ibrahim Time Domain (ITD) and LSCEM
scripts/
  run_denoising_demo.py      single-run denoising + SNR gain
  run_uncertainty_study.py   N-realization Monte-Carlo study (mean, u, U = 2u)
notebooks/        original Jupyter notebooks used in the study (outputs stripped)
results/          created when the study scripts are run (CSV summaries)
```

---

## What is reproduced here

* The **linear 3-DOF mass-spring-damper benchmark** (impulse response, positions and
  accelerations) and the additive-noise models used in the paper.
* The **TWN2N method** itself (time-window blind-spot dataset, autoencoder, L1 training
  with early stopping, inference).
* **Modal identification** with the Ibrahim Time Domain method, applied to noisy and to
  denoised signals.
* The **Monte-Carlo robustness study**: `N` independent realizations per noise level,
  reporting the ensemble mean, the standard uncertainty `u` and the expanded
  uncertainty `U = 2u` (coverage factor k = 2) of the identified natural frequencies.

The **nonlinear Hunt–Crossley impact** case study and the comparisons against the
Savitzky–Golay and wavelet (VisuShrink) filters are contained in the original
notebooks under `notebooks/`.

---

## Citation

If you use this code, please cite the manuscript above.

## License

MIT — see [LICENSE](LICENSE).
