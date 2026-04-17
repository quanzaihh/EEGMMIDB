"""Run train_n_r_dfbsa.train_network many times with different random seeds
and aggregate the best-validation-accuracy across runs (mean / std / variance).

Usage:
    python run_seeds_dfbsa.py                    # default: 50 runs, seeds 0..49, 50 epochs
    python run_seeds_dfbsa.py --runs 10 --epochs 20
    python run_seeds_dfbsa.py --seeds 100 101 102 --epochs 30
    python run_seeds_dfbsa.py --resume           # skip seeds already recorded in the CSV

Each run's best validation accuracy (max over epochs) is logged to
    training_outputs_LR/seed_runs/results.csv
and a final summary is written to
    training_outputs_LR/seed_runs/summary.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from torchvision import transforms

from dataset import ToTensor, EEGDataset2DLeftRight, EEGAugmentor  # type: ignore
from train_n_r_dfbsa import train_network, logger
from snn_n_r_dfbsa import WrapCUBASpikingCNN


# ---------- defaults mirroring train_n_r_dfbsa.__main__ ----------
SPIKE_TS = 160
BATCH_SIZE = 64
WT_LR = 0.0001
TS_LR = 0.0001
NEURON_LR = 0.0001
PARAM_LIST = [0.1, 0.1, 0.1, 0.3, 0.01, 0.1, 0.01]

DS_PARAMS = {
    "base_route": "../utils/eegmmidb_slice_norm/",
    "subject_id_list": [i + 1 for i in range(109) if not (i + 1 in [88, 89, 92, 100, 104, 106])],
    "start_ts": 0,
    "end_ts": 161,
    "window_ts": 160,
    "overlap_ts": 0,
    "use_imagery": False,
    "transform": ToTensor(),
}

VAL_LIST_DEFAULT = [i + 1 for i in range(10)]  # same as val_list[0] in the original script

OUT_DIR = Path.cwd() / "training_outputs_LR" / "seed_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "results.csv"
SUMMARY_TXT = OUT_DIR / "summary.txt"
SUMMARY_JSON = OUT_DIR / "summary.json"


def load_done_seeds(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["seed"]))
            except Exception:
                continue
    return done


def append_result_row(csv_path: Path, row: dict, header: list[str]):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summarize(csv_path: Path) -> dict:
    best_accs: list[float] = []
    final_accs: list[float] = []
    seeds: list[int] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                best_accs.append(float(row["best_acc"]))
                final_accs.append(float(row["final_acc"]))
                seeds.append(int(row["seed"]))
            except Exception:
                continue

    if not best_accs:
        return {"n": 0}

    best_arr = np.asarray(best_accs, dtype=np.float64)
    final_arr = np.asarray(final_accs, dtype=np.float64)
    n = int(best_arr.size)

    def stats(arr: np.ndarray) -> dict:
        mean = float(arr.mean())
        # sample variance / std (ddof=1) — more appropriate for "estimate across runs"
        var = float(arr.var(ddof=1)) if n > 1 else 0.0
        std = float(math.sqrt(var))
        sem = std / math.sqrt(n) if n > 0 else 0.0
        # 95% CI (normal approx, ~1.96 * SEM)
        ci95 = 1.96 * sem
        return {
            "n": n,
            "mean": mean,
            "variance": var,
            "std": std,
            "sem": sem,
            "ci95_halfwidth": ci95,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }

    return {
        "n_runs": n,
        "seeds": seeds,
        "best_acc": stats(best_arr),
        "final_acc": stats(final_arr),
    }


def write_summary(summary: dict):
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append(f"Total runs: {summary.get('n_runs', 0)}")
    lines.append(f"Seeds: {summary.get('seeds', [])}")
    lines.append("")
    for key in ("best_acc", "final_acc"):
        st = summary.get(key, {})
        if not st:
            continue
        lines.append(f"=== {key} (over {st.get('n', 0)} runs) ===")
        lines.append(f"  mean     : {st['mean']:.6f}  ({st['mean']*100:.3f} %)")
        lines.append(f"  variance : {st['variance']:.6e}")
        lines.append(f"  std      : {st['std']:.6f}  ({st['std']*100:.3f} pp)")
        lines.append(f"  sem      : {st['sem']:.6f}")
        lines.append(f"  95% CI ± : {st['ci95_halfwidth']:.6f}  ({st['ci95_halfwidth']*100:.3f} pp)")
        lines.append(f"  min      : {st['min']:.6f}")
        lines.append(f"  max      : {st['max']:.6f}")
        lines.append(f"  median   : {st['median']:.6f}")
        lines.append("")
    SUMMARY_TXT.write_text("\n".join(lines))
    logger.info("Wrote summary to %s", SUMMARY_TXT)
    print("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=50,
                   help="number of seeds to run (default: 50). Ignored if --seeds given.")
    p.add_argument("--seed-start", type=int, default=0,
                   help="starting seed (default: 0). Seeds will be seed-start..seed-start+runs-1.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="explicit list of seeds (overrides --runs/--seed-start).")
    p.add_argument("--epochs", type=int, default=50, help="epochs per run (default: 50).")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--scheduler", type=str, default="plateau",
                   choices=["plateau", "onecycle", "none"])
    p.add_argument("--validate-subject-list", type=int, nargs="+", default=None,
                   help="validation subject IDs (default: [1..10]).")
    p.add_argument("--resume", action="store_true",
                   help="skip seeds already present in results.csv.")
    p.add_argument("--summary-only", action="store_true",
                   help="do not train; just recompute summary from existing results.csv.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.summary_only:
        summary = summarize(RESULTS_CSV)
        write_summary(summary)
        return

    if args.seeds:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.runs))

    validate_subject_list = args.validate_subject_list or VAL_LIST_DEFAULT

    sched = None if args.scheduler == "none" else args.scheduler

    done = load_done_seeds(RESULTS_CSV) if args.resume else set()
    if done:
        logger.info("Resume: %d seeds already recorded, will skip.", len(done))

    header = ["seed", "best_acc", "final_acc", "best_epoch", "n_epochs",
              "elapsed_sec", "timestamp"]

    logger.info("Starting seed sweep: %d seeds, %d epochs each", len(seeds), args.epochs)
    t_all_start = time.time()

    for idx, seed in enumerate(seeds):
        if seed in done:
            logger.info("[%d/%d] seed=%d already done, skipping.", idx + 1, len(seeds), seed)
            continue

        logger.info("==================================================")
        logger.info("[%d/%d] Training with seed=%d", idx + 1, len(seeds), seed)
        logger.info("==================================================")

        # Fresh transform objects per run so internal randomness re-seeds
        ds_params = dict(DS_PARAMS)
        ds_params["transform"] = ToTensor()

        t0 = time.time()
        try:
            _, metrics = train_network(
                dataset=EEGDataset2DLeftRight,
                network=WrapCUBASpikingCNN,
                dataset_kwargs=ds_params,
                spike_ts=SPIKE_TS,
                param_list=PARAM_LIST,
                validate_subject_list=validate_subject_list,
                lr=[NEURON_LR, TS_LR, WT_LR],
                weight_decays=[2e-6, 4e-6, 1e-6],
                batch_size=args.batch_size,
                epoch=args.epochs,
                record_neuron=(0, 0, 0),
                seed=int(seed),
                scheduler_type=sched if sched is not None else "none",
            )
        except Exception:
            logger.exception("Run with seed=%d FAILED", seed)
            traceback.print_exc()
            continue
        elapsed = time.time() - t0

        epoch_accs = metrics.get("epoch_accs", []) or [0.0]
        best_idx = int(np.argmax(epoch_accs))
        best_acc = float(epoch_accs[best_idx])
        final_acc = float(epoch_accs[-1])

        row = {
            "seed": int(seed),
            "best_acc": f"{best_acc:.6f}",
            "final_acc": f"{final_acc:.6f}",
            "best_epoch": int(best_idx),
            "n_epochs": int(len(epoch_accs)),
            "elapsed_sec": f"{elapsed:.1f}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_result_row(RESULTS_CSV, row, header)
        logger.info("seed=%d done: best_acc=%.4f (epoch %d), final_acc=%.4f, elapsed=%.1fs",
                    seed, best_acc, best_idx, final_acc, elapsed)

        # incremental summary after each run
        try:
            summary = summarize(RESULTS_CSV)
            write_summary(summary)
        except Exception:
            logger.exception("Failed to update incremental summary")

    logger.info("All seeds finished in %.1f s", time.time() - t_all_start)
    summary = summarize(RESULTS_CSV)
    write_summary(summary)


if __name__ == "__main__":
    sys.exit(main() or 0)
