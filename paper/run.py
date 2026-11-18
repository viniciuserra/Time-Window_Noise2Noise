"""Run the case studies of the paper and store one JSON file per trial.

Examples (from the root of the repository)::

    python paper/run.py vibration --noise white --snr 15 --trials 0        # one trial, a few minutes
    python paper/run.py vibration --study white --trials 0-29 --workers 8  # white-noise sweep
    python paper/run.py vibration --study noise-types --trials 0-29 --workers 8
    python paper/run.py impact --pct 10 --trials 0
    python paper/run.py impact --study all --trials 0-2 --workers 5

Results go to ``paper/results/trials``. Trials that already have a result
file are skipped, so an interrupted run can be resumed with the same command.
Afterwards, ``python paper/summarize.py`` builds the tables and compares them
with the values reported in the paper.
"""
import argparse
import json
import multiprocessing
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results", "trials")


def tag(job):
    if job["case"] == "vibration":
        return f"vibration_{job['damping']}_{job['noise']}_snr{int(round(job['snr_db']))}_t{job['trial']:02d}"
    return f"impact_pct{int(job['noise_pct'])}_t{job['trial']:02d}"


def run_job(job, out_dir, threads, epochs):
    """Run one trial in the current process and write its JSON file."""
    for name in ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        os.environ.setdefault(name, str(threads))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")      # small dense networks train faster on the CPU
    for path in (ROOT, HERE):
        if path not in sys.path:
            sys.path.insert(0, path)
    import cases

    if job["case"] == "vibration":
        result = cases.vibration_case(job["noise"], job["snr_db"], job["trial"], job["damping"], epochs=epochs)
    else:
        result = cases.impact_case(job["noise_pct"], job["trial"], epochs=epochs)
    result["environment"] = _environment()
    path = os.path.join(out_dir, tag(job) + ".json")
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
    os.replace(path + ".tmp", path)
    return tag(job), result


def _environment():
    import platform

    import numpy
    import tensorflow

    import twn2n
    return dict(twn2n=twn2n.__version__, tensorflow=tensorflow.__version__, numpy=numpy.__version__,
                python=platform.python_version(), platform=platform.system())


def build_jobs(args):
    trials = _parse_trials(args.trials)
    jobs = []
    if args.case == "vibration":
        if args.study == "white":
            conditions = [("white", snr) for snr in (25.0, 20.0, 15.0, 12.0, 10.0, 5.0)]
        elif args.study == "noise-types":
            conditions = [(noise, 15.0) for noise in ("pink", "brown", "quantization")]
        elif args.study == "all":
            conditions = ([("white", snr) for snr in (25.0, 20.0, 15.0, 12.0, 10.0, 5.0)]
                          + [(noise, 15.0) for noise in ("pink", "brown", "quantization")])
        else:
            conditions = [(args.noise, args.snr)]
        for noise, snr in conditions:
            for trial in trials:
                jobs.append(dict(case="vibration", noise=noise, snr_db=float(snr), trial=trial,
                                 damping=args.damping))
    else:
        levels = (3, 5, 7, 10, 15) if args.study == "all" else (args.pct,)
        for pct in levels:
            for trial in trials:
                jobs.append(dict(case="impact", noise_pct=pct, trial=trial))
    return jobs


def _parse_trials(text):
    trials = []
    for part in str(text).split(","):
        if "-" in part:
            first, last = part.split("-")
            trials.extend(range(int(first), int(last) + 1))
        else:
            trials.append(int(part))
    return trials


def _line(name, result):
    if result["case"] == "vibration":
        return (f"[{name}] input {result['noisy_snr']:.2f} dB | TWN2N {result['twn2n_snr']:.2f} dB | "
                f"low-pass {result['lowpass_snr']:.2f} dB | {result['epochs_run']} epochs, "
                f"{result['training_time_s'] / 60:.1f} min")
    return (f"[{name}] distance {result['noisy_distance']:.4f} -> {result['twn2n_distance']:.4f} | "
            f"improved {result['impacts_improved']}/{result['n_impacts']} | {result['epochs_run']} epochs, "
            f"{result['training_time_s'] / 60:.1f} min")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", choices=("vibration", "impact"))
    parser.add_argument("--study", choices=("single", "white", "noise-types", "all"), default="single",
                        help="predefined set of conditions; 'single' uses --noise/--snr or --pct")
    parser.add_argument("--noise", default="white", choices=("white", "pink", "brown", "quantization"))
    parser.add_argument("--snr", type=float, default=15.0, help="input SNR in dB (vibration case)")
    parser.add_argument("--damping", default="R010", help="damping configuration of the 3-DOF system")
    parser.add_argument("--pct", type=int, default=10, help="noise level in %% of the signal std (impact case)")
    parser.add_argument("--trials", default="0", help="trial indices, for example 0, 0-29, or 0,3,7")
    parser.add_argument("--workers", type=int, default=1, help="number of parallel processes")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads per process")
    parser.add_argument("--epochs", type=int, default=5000, help="maximum number of epochs")
    parser.add_argument("--out", default=RESULTS, help="output folder")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs = [job for job in build_jobs(args) if not os.path.exists(os.path.join(args.out, tag(job) + ".json"))]
    print(f"{len(jobs)} trial(s) to run, {args.workers} worker(s)", flush=True)
    if len(jobs) == 1:
        print(_line(*run_job(jobs[0], args.out, args.threads, args.epochs)), flush=True)
        return
    # one fresh process per trial: every training starts from the same interpreter state
    tasks = [(job, args.out, args.threads, args.epochs) for job in jobs]
    with multiprocessing.get_context("spawn").Pool(max(1, args.workers), maxtasksperchild=1) as pool:
        for name, result, error in pool.imap_unordered(_safe_job, tasks):
            print(_line(name, result) if error is None else f"[{name}] FAILED: {error}", flush=True)


def _safe_job(task):
    job = task[0]
    try:
        name, result = run_job(*task)
        return name, result, None
    except Exception as error:                                       # keep the other trials running
        return tag(job), None, repr(error)


if __name__ == "__main__":
    main()
