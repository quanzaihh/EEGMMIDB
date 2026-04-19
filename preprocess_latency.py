"""
preprocess_latency.py
=====================

Measure CPU-side single-sample preprocessing latency for EEGMMIDB.

Two scenarios are measured *separately*, because they answer different questions
that reviewers sometimes conflate:

  Scenario A — pickle -> model-ready tensor (deployment hot path)
      EEGDataset2DLeftRight.__getitem__ reads the pre-sliced, pre-normalized
      pickle under eegmmidb_slice_norm/, optionally applies the EEGAugmentor,
      and produces the tensor the DataLoader feeds into the model.  This is
      the per-sample cost you pay during training/inference *once the dataset
      has been prepared*.

  Scenario B — raw .edf -> model-ready tensor (full pipeline)
      Starting from the original PhysioNet .edf file, we run the full utility.py
      chain:
          read_raw_data_n_events  (MNE load + firwin bandpass 0.1-79 Hz)
          slice_raw_data_between_events
          normalize_slice_raw_data  (z-score, channels)
          transform_slice_raw_data_2_2d  (1D 64-ch -> 10x11 2D topomap)
          epoch_2d_data_w_label  (windowing into fixed-length trials)
      This is the cost of the WHOLE preprocessing pipeline, from the file a
      BCI device actually produces.  It is typically 2-3 orders of magnitude
      slower than Scenario A, and that gap should be made explicit in the
      paper so reviewers do not mix the two numbers up.

Both scenarios report per-sample latency:
  - Scenario A: direct (one __getitem__ = one sample)
  - Scenario B: total end-to-end time / number of samples produced from the
    processed trial file (trials-to-samples ratio handled automatically).

Outputs
-------
power_latency_results/preprocess_latency.csv
power_latency_results/preprocess_latency_summary.txt

Usage
-----
    python preprocess_latency.py \
        --slice-route ../utils/eegmmidb_slice_norm/ \
        --raw-route   ../utils/eegmmidb/              # only needed if scenario B runs
        --subjects    1

If --raw-route is not provided or the edf files do not exist, Scenario B is
skipped with a clear message.
"""

import sys
import time
import csv
import json
import argparse
import statistics
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append("../utils/")

OUT_DIR = Path("power_latency_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Scenario A: __getitem__ latency on the pre-sliced pickle dataset
# ------------------------------------------------------------------
def measure_scenario_A(slice_route, subjects, n_warmup=100, n_measure=2000):
    """Time EEGDataset2DLeftRight.__getitem__."""
    from dataset import EEGDataset2DLeftRight, ToTensor, EEGAugmentor
    from torchvision.transforms import Compose

    # Standard config used by train_n_r_dfbsa.py
    ds_kwargs = dict(
        base_route=slice_route,
        subject_id_list=subjects,
        start_ts=0,
        end_ts=161,
        window_ts=160,
        overlap_ts=0,
        use_imagery=True,
    )

    # A1: raw __getitem__ (no transform)
    ds_notx = EEGDataset2DLeftRight(**ds_kwargs, transform=None)
    if len(ds_notx) == 0:
        raise RuntimeError(
            f"EEGDataset2DLeftRight produced 0 samples from {slice_route} "
            f"with subjects={subjects}. Check path and presence of .p files."
        )

    # A2: __getitem__ + ToTensor (the minimum the model expects)
    ds_tx = EEGDataset2DLeftRight(**ds_kwargs, transform=Compose([ToTensor()]))

    # A3: __getitem__ + Augmentor + ToTensor (training hot path)
    ds_aug = EEGDataset2DLeftRight(
        **ds_kwargs, transform=Compose([EEGAugmentor(), ToTensor()]))

    results = {}
    n = len(ds_notx)
    print(f"Scenario A dataset length: {n} samples")

    for tag, ds in [("raw_getitem", ds_notx),
                    ("getitem+ToTensor", ds_tx),
                    ("getitem+Aug+ToTensor", ds_aug)]:
        # warmup (also primes OS page cache for the pickle files)
        for i in range(min(n_warmup, n)):
            _ = ds[i % n]

        times_us = []
        for i in range(n_measure):
            idx = i % n
            t0 = time.perf_counter()
            _ = ds[idx]
            t1 = time.perf_counter()
            times_us.append((t1 - t0) * 1e6)

        arr = np.asarray(times_us)
        results[tag] = {
            "n": int(arr.size),
            "mean_us": float(arr.mean()),
            "std_us": float(arr.std(ddof=1)),
            "median_us": float(np.median(arr)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
            "min_us": float(arr.min()),
            "max_us": float(arr.max()),
        }
        print(
            f"  {tag:<22s} "
            f"mean={arr.mean():.1f} us  median={np.median(arr):.1f} us  "
            f"p95={np.percentile(arr, 95):.1f} us  p99={np.percentile(arr, 99):.1f} us"
        )

    return results


# ------------------------------------------------------------------
# Scenario B: raw .edf -> model-ready tensor (full pipeline)
# ------------------------------------------------------------------
def measure_scenario_B(raw_route, slice_route, subjects, n_reps=3):
    """Run full utility.py preprocessing pipeline end-to-end per trial file.

    Uses subject task_ids 4, 8, 12 (left/right motor imagery runs) which are
    the standard LR runs used by EEGDataset2DLeftRight.

    For each (subject, task) we time the complete pipeline and divide by the
    number of epoched samples produced, giving a per-sample preprocessing
    latency (incl. amortized EDF I/O + bandpass filter).
    """
    import utility

    # Load channel_mean / channel_std from the slice_route if present; otherwise
    # compute on the fly from a single subject. This mirrors what the training
    # code does: pre-computed stats are stored alongside the sliced pickles.
    mean_path = Path(slice_route) / "channel_mean.npy"
    std_path = Path(slice_route) / "channel_std.npy"
    if mean_path.is_file() and std_path.is_file():
        channel_mean = np.load(mean_path)
        channel_std = np.load(std_path)
    else:
        print(
            f"[scenario B] channel_mean/std not found in {slice_route}; "
            f"computing per-subject stats on the fly (still a fair pipeline "
            f"measurement, just adds a one-off cost not counted in the "
            f"per-sample average)."
        )
        channel_mean, channel_std = utility.compute_mean_std_for_all_data(
            raw_route, subjects)

    eeg_2d_map = utility.get_eeg_2d_map()
    map_2d_shape = (10, 11)

    # LR motor imagery tasks (left/right fist) per EEGMMIDB convention
    lr_task_ids = [4, 8, 12]

    rows = []
    aggregate_samples = 0
    aggregate_time_s = 0.0

    for subj in subjects:
        for task in lr_task_ids:
            edf_path = Path(raw_route) / f"S{subj:03d}" / f"S{subj:03d}R{task:02d}.edf"
            if not edf_path.is_file():
                print(f"[scenario B] missing {edf_path}, skipping")
                continue

            trial_times_s = []
            trial_samples = 0
            for _ in range(n_reps):
                t0 = time.perf_counter()
                # Step 1: MNE load + firwin bandpass
                raw_data, events = utility.read_raw_data_n_events(
                    str(Path(raw_route)) + "/", subj, task)
                # Step 2: slice between events
                slice_list, label_list = utility.slice_raw_data_between_events(
                    raw_data, events)
                # Step 3: z-score normalize
                norm_list = utility.normalize_slice_raw_data(
                    slice_list, channel_mean, channel_std)
                # Step 4: 1D (64 ch) -> 2D (10x11) topomap
                map_2d_list = utility.transform_slice_raw_data_2_2d(
                    norm_list, eeg_2d_map, map_2d_shape)
                # Step 5: window into fixed-length trials
                ep_data, ep_label, ep_ts = utility.epoch_2d_data_w_label(
                    map_2d_list, label_list,
                    start_ts=0, end_ts=161, window_ts=160, overlap_ts=0)
                t1 = time.perf_counter()
                trial_times_s.append(t1 - t0)
                trial_samples = int(ep_data.shape[0])

            avg_time_s = sum(trial_times_s) / len(trial_times_s)
            per_sample_ms = avg_time_s / trial_samples * 1000.0 if trial_samples > 0 else float("nan")
            print(
                f"  S{subj:03d}R{task:02d}: total={avg_time_s*1000:.1f} ms  "
                f"samples={trial_samples}  per-sample={per_sample_ms:.3f} ms"
            )
            rows.append({
                "subject": subj,
                "task": task,
                "total_ms": avg_time_s * 1000.0,
                "num_samples": trial_samples,
                "per_sample_ms": per_sample_ms,
                "n_reps": n_reps,
            })
            aggregate_samples += trial_samples
            aggregate_time_s += avg_time_s

    if not rows:
        return {"trials": [], "summary": None}

    overall_per_sample_ms = aggregate_time_s / aggregate_samples * 1000.0
    per_sample_list = [r["per_sample_ms"] for r in rows]
    summary = {
        "n_trials": len(rows),
        "aggregate_samples": aggregate_samples,
        "aggregate_time_s": aggregate_time_s,
        "overall_per_sample_ms": overall_per_sample_ms,
        "trial_per_sample_mean_ms": float(statistics.mean(per_sample_list)),
        "trial_per_sample_std_ms": (float(statistics.stdev(per_sample_list))
                                    if len(per_sample_list) > 1 else 0.0),
    }
    print(
        f"\n  overall (amortized): {overall_per_sample_ms:.3f} ms/sample "
        f"across {aggregate_samples} samples from {len(rows)} trial files"
    )
    return {"trials": rows, "summary": summary}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slice-route", default="../utils/eegmmidb_slice_norm/",
                   help="Path to pre-sliced/normalized pickle dataset (Scenario A).")
    p.add_argument("--raw-route", default="",
                   help="Path to raw PhysioNet .edf tree (Scenario B). Leave "
                        "empty to skip scenario B.")
    p.add_argument("--subjects", type=int, nargs="+", default=[7, 38],
                   help="Subject IDs to use. Default 7, 38 (have pickles in "
                        "eegmmidb_slice_norm/).")
    p.add_argument("--n-warmup", type=int, default=100)
    p.add_argument("--n-measure", type=int, default=2000)
    p.add_argument("--n-reps-B", type=int, default=3,
                   help="Number of repetitions for each (subject,task) in "
                        "Scenario B.")
    return p.parse_args()


def main():
    args = parse_args()

    banner_lines = [
        f"python      : {sys.version.split()[0]}",
        f"numpy       : {np.__version__}",
        f"slice_route : {args.slice_route}",
        f"raw_route   : {args.raw_route or '(scenario B skipped)'}",
        f"subjects    : {args.subjects}",
    ]
    banner = "\n".join(banner_lines)
    print(banner)
    print()

    print("=" * 70)
    print("Scenario A: pickle -> model-ready tensor (deployment hot path)")
    print("=" * 70)
    try:
        result_A = measure_scenario_A(
            args.slice_route, args.subjects,
            n_warmup=args.n_warmup, n_measure=args.n_measure)
    except Exception as e:
        print(f"[scenario A] FAILED: {type(e).__name__}: {e}")
        result_A = None
    print()

    print("=" * 70)
    print("Scenario B: raw .edf -> model-ready tensor (full pipeline)")
    print("=" * 70)
    if not args.raw_route:
        print("[scenario B] --raw-route not given, skipping.")
        result_B = None
    else:
        try:
            result_B = measure_scenario_B(
                args.raw_route, args.slice_route, args.subjects,
                n_reps=args.n_reps_B)
        except Exception as e:
            print(f"[scenario B] FAILED: {type(e).__name__}: {e}")
            result_B = None
    print()

    # ---- write artifacts ----
    csv_path = OUT_DIR / "preprocess_latency.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "variant", "metric", "value"])
        if result_A:
            for variant, d in result_A.items():
                for k, v in d.items():
                    w.writerow(["A", variant, k, v])
        if result_B and result_B["summary"]:
            for trial in result_B["trials"]:
                for k, v in trial.items():
                    w.writerow(["B_trial", f"S{trial['subject']:03d}R{trial['task']:02d}", k, v])
            for k, v in result_B["summary"].items():
                w.writerow(["B_summary", "overall", k, v])

    summary_path = OUT_DIR / "preprocess_latency_summary.txt"
    with open(summary_path, "w") as f:
        f.write(banner + "\n\n")
        payload = {
            "scenario_A_pickle_to_tensor_us": result_A,
            "scenario_B_edf_to_tensor_ms": result_B,
            "config": {
                "slice_route": args.slice_route,
                "raw_route": args.raw_route,
                "subjects": args.subjects,
                "n_warmup": args.n_warmup,
                "n_measure": args.n_measure,
                "n_reps_B": args.n_reps_B,
            },
        }
        f.write(json.dumps(payload, indent=2, default=str))

    print(f"Artifacts written to: {OUT_DIR}/")
    if result_A and result_B and result_B.get("summary"):
        a_us = result_A["getitem+ToTensor"]["mean_us"]
        b_ms = result_B["summary"]["overall_per_sample_ms"]
        print(
            f"\nHeadline comparison:\n"
            f"  Scenario A (pickle + ToTensor)  = {a_us:.1f} us/sample\n"
            f"  Scenario B (raw edf -> tensor)  = {b_ms:.3f} ms/sample\n"
            f"  Ratio B/A                        = {b_ms * 1000 / a_us:.0f}x"
        )


if __name__ == "__main__":
    main()
