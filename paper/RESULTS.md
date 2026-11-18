# Case studies of the paper, run with the library

This file is written by `paper/summarize.py`. It compares the trials run with this library (`paper/run.py`, stored in `paper/results/`) with the values reported in the paper (`paper/reference/`). Every trial uses the noise realization of the corresponding trial of the paper. The result tables of the paper themselves are in `paper/TABLES.md`.

## Vibration of the 3-DOF system

9 trials run with the library (Python 3.11.9, TensorFlow 2.21.0, NumPy 2.4.6, Windows). Training took from 1.2 to 3.6 min per trial (138 to 573 epochs).

### TWN2N output SNR

SNR of the displacements after denoising, in dB. The paper column gives the mean ± standard deviation of its thirty trials, with the range in parentheses. The library column gives trial 0. The output of TWN2N depends on the initial weights of the network, so a trial of the library is expected to fall inside the range of the paper, not to repeat one of its trials.

| Noise | Input SNR | Output, paper | Output, library | Mean gain, paper | Gain, library |
|---|---|---|---|---|---|
| white | 25.00 | 24.04 ± 1.04 (21.67 to 26.19) | 25.21 | -0.96 | +0.21 |
| white | 20.00 | 23.11 ± 0.78 (21.58 to 24.91) | 23.86 | +3.11 | +3.86 |
| white | 15.00 | 21.01 ± 0.68 (19.20 to 22.12) | 20.92 | +6.01 | +5.92 |
| white | 12.00 | 19.54 ± 0.57 (17.80 to 20.50) | 19.29 | +7.54 | +7.29 |
| white | 10.00 | 18.09 ± 0.69 (16.16 to 18.91) | 18.71 | +8.10 | +8.71 |
| white | 5.00 | 13.87 ± 0.64 (12.68 to 15.26) | 13.50 | +8.88 | +8.50 |
| pink | 15.00 | 17.48 ± 0.66 (16.05 to 18.84) | 17.61 | +2.48 | +2.61 |
| brown | 15.00 | 15.28 ± 0.50 (14.63 to 16.61) | 15.18 | +0.28 | +0.18 |
| quantization | 15.36 | 15.73 ± 0.28 (15.07 to 16.21) | 15.67 | +0.37 | +0.31 |

### Classical filters

SNR gain of the displacements, in dB, for the same trials in the paper and in the library: zero-phase Butterworth low-pass filter of fourth order with the cutoff fixed at 1.05 f3; Savitzky-Golay filter (window of five samples, third order); and VisuShrink with the exact noise level. These filters involve no training, so the two columns of each filter must coincide.

| Noise | Input SNR | Low-pass, paper | Low-pass, library | Savitzky-Golay, paper | Savitzky-Golay, library | VisuShrink, paper | VisuShrink, library |
|---|---|---|---|---|---|---|---|
| white | 25.00 | +2.01 | +2.01 | +2.25 | +2.25 | -2.81 | -2.81 |
| white | 20.00 | +2.89 | +2.89 | +2.81 | +2.81 | -1.70 | -1.70 |
| white | 15.00 | +3.22 | +3.22 | +3.00 | +3.00 | -0.45 | -0.45 |
| white | 12.00 | +3.32 | +3.32 | +3.06 | +3.06 | +0.55 | +0.55 |
| white | 10.00 | +3.38 | +3.38 | +3.10 | +3.10 | +1.12 | +1.12 |
| white | 5.00 | +3.41 | +3.41 | +3.12 | +3.12 | +2.88 | +2.88 |
| pink | 15.00 | +0.35 | +0.35 | +0.38 | +0.38 | -1.89 | -1.89 |
| brown | 15.00 | -0.07 | -0.07 | -0.05 | -0.05 | -3.08 | -3.08 |
| quantization | 15.36 | +1.80 | +1.80 | +1.76 | +1.76 | -0.80 | -0.80 |

### Gain inside the structural band

SNR gain of the displacements restricted to the band from 0 to 1.25 f3, in dB. The TWN2N column of the paper gives the mean ± standard deviation and the range of its thirty trials; the low-pass columns compare the same trials.

| Noise | Input SNR | TWN2N, paper | TWN2N, library | Low-pass, paper | Low-pass, library |
|---|---|---|---|---|---|
| white | 25.00 | -0.14 ± 1.35 (-3.61 to 2.85) | 0.97 | +0.67 | +0.67 |
| white | 20.00 | 3.28 ± 0.91 (1.43 to 5.02) | 3.91 | +0.97 | +0.97 |
| white | 15.00 | 5.67 ± 0.82 (3.91 to 6.98) | 5.27 | +1.03 | +1.03 |
| white | 12.00 | 6.80 ± 0.66 (5.49 to 8.14) | 7.13 | +1.06 | +1.06 |
| white | 10.00 | 7.40 ± 0.76 (5.77 to 8.56) | 8.18 | +1.10 | +1.10 |
| white | 5.00 | 8.16 ± 0.54 (6.80 to 9.15) | 8.25 | +1.10 | +1.10 |
| pink | 15.00 | 2.86 ± 0.77 (1.41 to 4.38) | 3.06 | +0.13 | +0.13 |
| brown | 15.00 | 0.49 ± 0.52 (-0.12 to 1.75) | 0.42 | -0.02 | -0.02 |
| quantization | 15.36 | 0.32 ± 0.20 (0.01 to 0.68) | 0.25 | +0.68 | +0.68 |

### Damping ratios identified by the ITD method

Damping ratio of each mode identified from the displacements of the first record. The reference values are 0.0100, 0.0072, and 0.0100. The noisy columns compare the same trials and must coincide, because the identification from the noisy signals involves no training. The TWN2N column of the paper gives the mean ± standard deviation of its thirty trials.

| Input SNR | Mode | Noisy, paper | Noisy, library | TWN2N, paper | TWN2N, library |
|---|---|---|---|---|---|
| 20 | 1 | 0.0102 | 0.0102 | 0.0094 ± 0.0005 | 0.0095 |
| 20 | 2 | 0.0075 | 0.0075 | 0.0065 ± 0.0004 | 0.0067 |
| 20 | 3 | 0.0650 | 0.0650 | 0.0093 ± 0.0007 | 0.0101 |
| 15 | 1 | 0.0106 | 0.0106 | 0.0095 ± 0.0004 | 0.0103 |
| 15 | 2 | 0.0084 | 0.0084 | 0.0081 ± 0.0102 | 0.0064 |
| 15 | 3 | 0.0742 | 0.0742 | 0.0099 ± 0.0011 | 0.0086 |

## Nonlinear impact (Hunt and Crossley)

5 trials run with the library (Python 3.11.9, TensorFlow 2.21.0, NumPy 2.4.6, Windows). Training took from 38 to 53 min per trial.

### Distance to the clean state

Dimensionless distance between the estimated and the clean state (indentation, velocity, and force, each divided by the standard deviation of its clean signal), averaged over the twenty impacts. The TWN2N column of the paper gives the mean ± standard deviation of its three trials, with the range in parentheses; the library columns give trial 0.

| Input SNR | Noisy, paper | Noisy, library | TWN2N, paper | TWN2N, library |
|---|---|---|---|---|
| 30.46 | 0.024 | 0.024 | 0.0129 ± 0.0005 (0.0122 to 0.0133) | 0.0126 |
| 26.02 | 0.040 | 0.040 | 0.0192 ± 0.0013 (0.0180 to 0.0206) | 0.0173 |
| 23.10 | 0.056 | 0.056 | 0.0234 ± 0.0012 (0.0220 to 0.0243) | 0.0210 |
| 20.00 | 0.080 | 0.080 | 0.0293 ± 0.0014 (0.0285 to 0.0310) | 0.0294 |
| 16.48 | 0.120 | 0.119 | 0.0415 ± 0.0012 (0.0402 to 0.0425) | 0.0401 |

### Error per variable after denoising

Normalized root-mean-square error of the TWN2N output: indentation, velocity, measured force, and force recomputed from the kinematic signals with the Hunt and Crossley law. The paper columns give the mean of its three trials.

| Input SNR | x, paper | x, library | v, paper | v, library | F, paper | F, library | F (law), paper | F (law), library |
|---|---|---|---|---|---|---|---|---|
| 30.46 | 0.014 | 0.013 | 0.012 | 0.011 | 0.012 | 0.013 | 0.012 | 0.011 |
| 26.02 | 0.020 | 0.017 | 0.018 | 0.017 | 0.018 | 0.017 | 0.018 | 0.014 |
| 23.10 | 0.025 | 0.023 | 0.023 | 0.020 | 0.022 | 0.020 | 0.022 | 0.020 |
| 20.00 | 0.032 | 0.031 | 0.028 | 0.025 | 0.027 | 0.031 | 0.027 | 0.027 |
| 16.48 | 0.045 | 0.043 | 0.038 | 0.037 | 0.040 | 0.040 | 0.040 | 0.037 |
