"""Command-line use of the library: denoise the signals stored in a CSV file.

    python -m twn2n noisy.csv denoised.csv --p 2 --r 4

The input file has one row per time sample and one column per channel, with or
without a header line. Columns that are not signals (for example, a time
column) are excluded with ``--columns`` and copied unchanged to the output.
"""
import argparse
import sys

import numpy as np

from .denoiser import TWN2N
from .network import NETWORKS


def read_table(path, delimiter=","):
    """Read a numeric CSV file. Returns ``(header or None, data)`` with ``data`` two-dimensional."""
    with open(path, encoding="utf-8-sig") as handle:
        first = handle.readline()
    try:
        [float(value) for value in first.strip().split(delimiter)]
        header = None
    except ValueError:
        header = first.strip()
    data = np.loadtxt(path, delimiter=delimiter, skiprows=0 if header is None else 1, ndmin=2,
                      encoding="utf-8-sig")
    return header, data


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m twn2n",
                                     description="Denoise the signals of a CSV file with TWN2N.")
    parser.add_argument("input", help="CSV file with one row per time sample and one column per channel")
    parser.add_argument("output", help="CSV file that receives the denoised signals")
    parser.add_argument("--p", type=int, default=2, help="neighbors on each side of the masked sample (default: 2)")
    parser.add_argument("--r", type=int, required=True,
                        help="units of the bottleneck, smaller than the number of channels")
    parser.add_argument("--columns", default=None,
                        help="indices of the signal columns, starting at 0 and separated by commas (default: all)")
    parser.add_argument("--network", default="standard", choices=NETWORKS, help="architecture (default: standard)")
    parser.add_argument("--epochs", type=int, default=5000, help="maximum number of epochs (default: 5000)")
    parser.add_argument("--seed", type=int, default=None, help="seed of the training")
    parser.add_argument("--delimiter", default=",", help="column separator (default: comma)")
    parser.add_argument("--save-model", default=None, help="file (.npz) that receives the trained network")
    parser.add_argument("--verbose", type=int, default=0, choices=(0, 1, 2), help="training messages (default: 0)")
    args = parser.parse_args(argv)

    header, data = read_table(args.input, args.delimiter)
    columns = list(range(data.shape[1])) if args.columns is None else [int(c) for c in args.columns.split(",")]
    print(f"{data.shape[0]} samples, {len(columns)} channel(s); training...", flush=True)

    model = TWN2N(p=args.p, r=args.r, network=args.network, epochs=args.epochs,
                  seed=args.seed, verbose=args.verbose)
    result = data.copy()
    result[:, columns] = model.fit_denoise(data[:, columns]).reshape(len(data), len(columns))
    np.savetxt(args.output, result, delimiter=args.delimiter, header=header or "", comments="")
    if args.save_model:
        model.save(args.save_model)
    print(f"done: {model.epochs_run_} epochs (best: {model.best_epoch_}), {model.training_time_s_:.0f} s, "
          f"bottleneck of {model.r_} unit(s). Denoised signals written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
