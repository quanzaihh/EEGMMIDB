"""
latency_benchmark.py
====================

Measure GPU inference latency for WrapCUBASpikingCNN (from snn_n_r_dfbsa.py).

What it reports
---------------
1. End-to-end per-sample latency (batch=1, 500 runs): mean / median / p95 / p99.
2. Batch=64 throughput (samples/s, ms/sample).
3. Per-module CUDA time breakdown via torch.profiler, aggregated by top-level
   sub-module name (conv1, cbam_conv1, conv2, cbam_conv2, avg_pool, conv3,
   cbam_conv3, temp_conv1, fc1, fc2).

Notes
-----
- Uses RANDOM weights. Latency is a function of the forward computation graph,
  not weight values; this is a valid measurement without a trained checkpoint.
  SynOps / energy estimates that DO depend on spike statistics are produced by
  synops_energy.py (second batch, requires a checkpoint).
- snn_n_r_dfbsa.CUBASpikingCNN.forward hard-codes torch.device("cuda") for its
  dropout masks, so this script REQUIRES CUDA.
- GPU model / driver / PyTorch version are printed at the top so readers can
  reproduce.

Outputs
-------
power_latency_results/latency_summary.txt
power_latency_results/latency_per_module.csv
power_latency_results/latency_end_to_end.csv
"""

import sys
import time
import csv
import json
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append("../utils/")

from snn_n_r_dfbsa import WrapCUBASpikingCNN


OUT_DIR = Path("power_latency_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPIKE_TS = 160
PARAM_LIST = [0.1, 0.1, 0.1, 0.3, 0.01, 0.1, 0.01]
WARMUP_ITERS = 50
MEASURE_ITERS = 500
PROFILER_ITERS = 50
BATCH_SIZE_THROUGHPUT = 64

# Top-level submodule names we track in the per-module breakdown.
MODULE_BUCKETS = [
    "conv1", "cbam_conv1",
    "conv2", "cbam_conv2",
    "avg_pool",
    "conv3", "cbam_conv3",
    "temp_conv1",
    "fc1", "fc2",
]


def env_banner() -> str:
    lines = [
        f"torch              : {torch.__version__}",
        f"cuda available     : {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines += [
            f"cuda device        : {torch.cuda.get_device_name(0)}",
            f"cuda capability    : {torch.cuda.get_device_capability(0)}",
            f"cudnn version      : {torch.backends.cudnn.version()}",
        ]
    return "\n".join(lines)


def build_model(device):
    net = WrapCUBASpikingCNN(SPIKE_TS, device, param_list=PARAM_LIST, record_neuron=None)
    net = net.to(device)
    net.eval()
    return net


def make_dummy(batch_size, device):
    # Dataset __getitem__ produces [1, 10, 11, 160] per sample;
    # collated batch expected by model is [B, 1, 10, 11, 160].
    return torch.randn(batch_size, 1, 10, 11, SPIKE_TS, device=device)


@torch.no_grad()
def measure_end_to_end(net, device):
    """Batch=1 per-sample latency distribution."""
    dummy = make_dummy(1, device)

    for _ in range(WARMUP_ITERS):
        _ = net(dummy)
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(MEASURE_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = net(dummy)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(samples_ms)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "samples_ms": samples_ms,
    }


@torch.no_grad()
def measure_throughput(net, device, batch_size=BATCH_SIZE_THROUGHPUT, iters=100):
    dummy = make_dummy(batch_size, device)
    for _ in range(20):
        _ = net(dummy)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = net(dummy)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_samples = batch_size * iters
    dt = t1 - t0
    return {
        "batch_size": batch_size,
        "iters": iters,
        "total_samples": total_samples,
        "total_time_s": dt,
        "samples_per_s": total_samples / dt,
        "ms_per_batch": dt / iters * 1000.0,
        "ms_per_sample": dt / total_samples * 1000.0,
    }


@torch.no_grad()
def measure_per_module(net, device):
    """Run torch.profiler and group CUDA time by top-level submodule name.

    We wrap the forward of each top-level submodule in record_function, so
    every call produces a named span the profiler can aggregate by key.
    """
    from torch.profiler import profile, record_function, ProfilerActivity

    snn = net.snn
    patched = []

    def wrap(mod, name):
        orig = mod.forward

        def wrapped_forward(*args, **kwargs):
            with record_function(f"MOD::{name}"):
                return orig(*args, **kwargs)

        mod.forward = wrapped_forward
        patched.append((mod, orig))

    for name in MODULE_BUCKETS:
        if hasattr(snn, name):
            wrap(getattr(snn, name), name)

    dummy = make_dummy(1, device)

    for _ in range(10):
        _ = net(dummy)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(PROFILER_ITERS):
            _ = net(dummy)
            torch.cuda.synchronize()

    # restore forwards
    for mod, orig in patched:
        mod.forward = orig

    agg = {}
    for ev in prof.key_averages():
        k = ev.key
        if not k.startswith("MOD::"):
            continue
        name = k[len("MOD::"):]
        cuda_us = float(getattr(ev, "cuda_time_total", 0.0))
        cpu_us = float(getattr(ev, "cpu_time_total", 0.0))
        count = int(getattr(ev, "count", 0))
        agg[name] = {
            "cuda_time_us_total": cuda_us,
            "cpu_time_us_total": cpu_us,
            "count": count,
        }

    rows = []
    for name in MODULE_BUCKETS:
        if name in agg:
            v = agg[name]
            per_sample_ms = v["cuda_time_us_total"] / 1000.0 / PROFILER_ITERS
            rows.append({
                "module": name,
                "count_total": v["count"],
                "cuda_ms_per_sample": per_sample_ms,
                "cuda_percent": 0.0,
            })

    module_total_ms = sum(r["cuda_ms_per_sample"] for r in rows)
    for r in rows:
        r["cuda_percent"] = (r["cuda_ms_per_sample"] / module_total_ms * 100.0) if module_total_ms > 0 else 0.0

    # Also compute CBAM aggregate vs non-CBAM aggregate (handy for the paper).
    cbam_names = {"cbam_conv1", "cbam_conv2", "cbam_conv3"}
    cbam_ms = sum(r["cuda_ms_per_sample"] for r in rows if r["module"] in cbam_names)
    non_cbam_ms = module_total_ms - cbam_ms
    split = {
        "cbam_total_ms_per_sample": cbam_ms,
        "non_cbam_total_ms_per_sample": non_cbam_ms,
        "cbam_percent_of_tracked": (cbam_ms / module_total_ms * 100.0) if module_total_ms > 0 else 0.0,
    }

    return rows, module_total_ms, split


def main():
    if not torch.cuda.is_available():
        print("[FATAL] CUDA is required (CUBASpikingCNN hard-codes torch.device('cuda')).")
        sys.exit(1)

    device = torch.device("cuda")
    banner = env_banner()
    print(banner)
    print()

    net = build_model(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Total params: {n_params:,}")
    print()

    print("=" * 70)
    print("[1/3] End-to-end per-sample latency (batch=1)")
    print("=" * 70)
    e2e = measure_end_to_end(net, device)
    print(
        f"  n={e2e['n']}  mean={e2e['mean_ms']:.3f} ms  std={e2e['std_ms']:.3f} ms\n"
        f"  median={e2e['median_ms']:.3f} ms  p95={e2e['p95_ms']:.3f} ms  "
        f"p99={e2e['p99_ms']:.3f} ms\n"
        f"  min={e2e['min_ms']:.3f} ms  max={e2e['max_ms']:.3f} ms"
    )
    print()

    print("=" * 70)
    print(f"[2/3] Throughput (batch={BATCH_SIZE_THROUGHPUT})")
    print("=" * 70)
    tp = measure_throughput(net, device)
    print(
        f"  total_samples={tp['total_samples']}  time={tp['total_time_s']:.2f} s\n"
        f"  throughput={tp['samples_per_s']:.1f} samples/s\n"
        f"  ms/batch={tp['ms_per_batch']:.3f}   ms/sample={tp['ms_per_sample']:.3f}"
    )
    print()

    print("=" * 70)
    print("[3/3] Per-module breakdown (batch=1, torch.profiler)")
    print("=" * 70)
    rows, module_total_ms, split = measure_per_module(net, device)
    print(f"{'module':<14} {'calls':>8} {'ms/sample':>12} {'%':>8}")
    for r in rows:
        print(f"{r['module']:<14} {r['count_total']:>8} "
              f"{r['cuda_ms_per_sample']:>12.4f} {r['cuda_percent']:>7.2f}%")
    print(f"{'TRACKED SUM':<14} {'':>8} {module_total_ms:>12.4f} {100.0:>7.2f}%")
    print()
    print(f"CBAM aggregate     : {split['cbam_total_ms_per_sample']:.4f} ms/sample "
          f"({split['cbam_percent_of_tracked']:.2f}% of tracked)")
    print(f"non-CBAM aggregate : {split['non_cbam_total_ms_per_sample']:.4f} ms/sample")
    print()
    print(f"Note: end-to-end per-sample latency = {e2e['mean_ms']:.3f} ms; "
          f"sum of tracked modules = {module_total_ms:.3f} ms. The gap is overhead "
          f"outside tracked submodules (python loop, view/reshape, ts_weights scaling, "
          f"dropout mask sampling, state init).")

    # ---- write artifacts ----
    with open(OUT_DIR / "latency_end_to_end.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_idx", "latency_ms"])
        for i, v in enumerate(e2e["samples_ms"]):
            w.writerow([i, f"{v:.6f}"])

    with open(OUT_DIR / "latency_per_module.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "count_total", "cuda_ms_per_sample", "cuda_percent"])
        for r in rows:
            w.writerow([r["module"], r["count_total"],
                        f"{r['cuda_ms_per_sample']:.6f}",
                        f"{r['cuda_percent']:.4f}"])
        w.writerow(["__tracked_sum__", "", f"{module_total_ms:.6f}", "100.0"])
        w.writerow(["__cbam_total__", "",
                    f"{split['cbam_total_ms_per_sample']:.6f}",
                    f"{split['cbam_percent_of_tracked']:.4f}"])
        w.writerow(["__non_cbam_total__", "",
                    f"{split['non_cbam_total_ms_per_sample']:.6f}", ""])

    summary = {
        "env": banner,
        "total_params": n_params,
        "end_to_end_batch1": {k: v for k, v in e2e.items() if k != "samples_ms"},
        "throughput_batch64": tp,
        "per_module_ms_per_sample": {r["module"]: r["cuda_ms_per_sample"] for r in rows},
        "per_module_percent": {r["module"]: r["cuda_percent"] for r in rows},
        "module_total_ms_per_sample": module_total_ms,
        "cbam_vs_non_cbam": split,
        "config": {
            "spike_ts": SPIKE_TS,
            "warmup_iters": WARMUP_ITERS,
            "measure_iters": MEASURE_ITERS,
            "profiler_iters": PROFILER_ITERS,
            "throughput_batch_size": BATCH_SIZE_THROUGHPUT,
            "param_list": PARAM_LIST,
        },
    }
    with open(OUT_DIR / "latency_summary.txt", "w") as f:
        f.write(banner + "\n\n")
        f.write(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
