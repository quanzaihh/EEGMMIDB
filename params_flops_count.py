"""
params_flops_count.py
=====================

Static analysis (no checkpoint required) of the WrapCUBASpikingCNN model
from snn_n_r_dfbsa.py, answering two questions:

  Q1. How many parameters and FLOPs does the FULL model have, broken down per
      top-level submodule (conv1 / cbam_conv1 / conv2 / ... / fc1 / fc2)?
  Q2. How much of that is contributed by the three CBAM blocks? I.e. what is
      the "attention cost" in purely structural terms?

Why this is valid without a checkpoint
--------------------------------------
Parameter counts are a function of layer shapes, not weight values. Dense
(ANN-style) FLOPs counted here are the *maximum* work per forward pass — the
work done if every spike fired every timestep. They are a legitimate upper
bound on SNN compute and are the standard "complexity" number reviewers ask
for alongside params.

A separate synops_energy.py (second batch, needs a checkpoint) will report the
actual SynOps = FLOPs × mean_spike_rate, which is what the hardware truly
runs.  For the narrative "attention module is small" the params / FLOPs delta
here is already a conclusive, reviewer-friendly answer.

What we count (per forward over `spike_ts` timesteps)
-----------------------------------------------------
  * conv1 / conv2 / conv3 : 2D convolution MACs × spike_ts (FeedForwardCUBALIFCell
    invokes its inner Conv2d once per timestep).
  * cbam_conv1/2/3 : per the CBAM forward (ChannelGate MLP + SpatialGate conv).
    Called once per timestep on the post-LIF spike tensor.
  * temp_conv1 : 3 nn.Linear(256,256) membranes, one of them evaluated per
    timestep (TemporalConvCUBALIFCell rotates through kernel_size=3).
  * fc1 : Linear(256,256) × spike_ts.
  * fc2 : Linear(256,2), executed ONCE at the end on the weighted spike sum.

Convention
----------
  - "MACs" = multiply-accumulate operations (1 MAC = 2 FLOPs for dense MAC, but
    a spike-driven SNN can execute 1 AC instead of 1 MAC). We report MACs so
    the reader can multiply by the energy constant they prefer.
  - We ignore biases and activations / pointwise ops (< 1% of total).
  - We ignore LIF state updates (cdecay/vdecay/vth/reset) which are per-neuron
    elementwise ops; see the "other_elementwise_ops" field.

Outputs
-------
power_latency_results/params_flops_table.csv
power_latency_results/params_flops_summary.md
"""

import sys
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent))

from snn_n_r_dfbsa import WrapCUBASpikingCNN


OUT_DIR = Path("power_latency_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPIKE_TS = 160
PARAM_LIST = [0.1, 0.1, 0.1, 0.3, 0.01, 0.1, 0.01]


# ------------------------------------------------------------------
# Shape bookkeeping (hard-coded from snn_n_r_dfbsa.CUBASpikingCNN)
# ------------------------------------------------------------------
# Input at each timestep: [B, 1, 10, 11]  (batch, channel, H, W)
#   conv1: Conv2d(1,64,(3,3))  -> [B, 64, 8, 9]     (H:10-3+1=8, W:11-3+1=9)
#   cbam_conv1 on                [B, 64, 8, 9]
#   conv2: Conv2d(64,128,(3,3))-> [B, 128, 6, 7]
#   cbam_conv2 on                [B, 128, 6, 7]
#   avg_pool 2x2                -> [B, 128, 3, 3]
#   conv3: Conv2d(128,256,(3,3))-> [B, 256, 1, 1]
#   cbam_conv3 on                [B, 256, 1, 1]
#   reshape to [B, 256]
#   temp_conv1: Linear(256,256) picks one of 3 per timestep
#   fc1: Linear(256,256)
#   (loop over spike_ts=160 timesteps)
#   fc2: Linear(256,2), executed once on accumulated spike sum
#
SHAPES = {
    "input":       (1, 10, 11),
    "after_conv1": (64, 8, 9),
    "after_conv2": (128, 6, 7),
    "after_pool":  (128, 3, 3),
    "after_conv3": (256, 1, 1),
}


# ------------------------------------------------------------------
# MAC counters
# ------------------------------------------------------------------
def conv2d_macs(in_c, out_c, kH, kW, out_H, out_W) -> int:
    """MACs per forward for a dense Conv2d (ignoring bias add)."""
    return in_c * out_c * kH * kW * out_H * out_W


def linear_macs(in_f, out_f) -> int:
    return in_f * out_f


def cbam_macs(channels: int, H: int, W: int, reduction: int = 16,
              spatial_k: int = 7) -> Dict[str, int]:
    """Return MACs breakdown for the CBAM block defined in cbam.py.

    Structure (from cbam.py):
      ChannelGate:
        avg_pool + max_pool -> [B, C]
        MLP = Linear(C, C//r) + ReLU + Linear(C//r, C)   # applied to BOTH pools
        sigmoid, elementwise multiply with input
      SpatialGate:
        ChannelPool: mean + max along channel axis -> [B, 2, H, W]
        Conv2d(2, 1, (spatial_k, spatial_k), padding=spatial_k//2)
        sigmoid, elementwise multiply with input

    We count only MACs (the dominant work). Pointwise ops (sigmoid, mul) are
    accumulated into "elementwise_ops".
    """
    red = max(channels // reduction, 1)

    # ChannelGate MLP: applied once for avg-pool output, once for max-pool
    mlp_macs = 2 * (linear_macs(channels, red) + linear_macs(red, channels))

    # SpatialGate conv: input shape is [B, 2, H, W], output [B, 1, H, W].
    # padding=spatial_k//2 is SAME padding -> out_H=H, out_W=W.
    sg_macs = conv2d_macs(2, 1, spatial_k, spatial_k, H, W)

    # Elementwise: avg+max pooling (C*H*W * 2), sigmoid (C+H*W), broadcast
    # multiplications (C*H*W * 2). These are << MACs; we tally them separately.
    ew = 4 * channels * H * W + channels + H * W

    return {
        "channel_gate_mlp_macs": mlp_macs,
        "spatial_gate_conv_macs": sg_macs,
        "macs_total": mlp_macs + sg_macs,
        "elementwise_ops": ew,
    }


# ------------------------------------------------------------------
# Param counts from the actual instantiated model
# ------------------------------------------------------------------
def count_params_by_module(net: nn.Module) -> Dict[str, int]:
    """Count trainable params for each top-level submodule of net.snn.

    We also aggregate nn.Parameter attributes (ts_weights, cdecay/vdecay
    tables) into a 'neuron_state' bucket so nothing is orphaned.
    """
    snn = net.snn
    tracked = [
        "conv1", "conv2", "conv3",
        "cbam_conv1", "cbam_conv2", "cbam_conv3",
        "avg_pool", "temp_conv1", "fc1", "fc2",
    ]
    by_mod = {}
    tracked_ids = set()

    for name in tracked:
        mod = getattr(snn, name, None)
        if mod is None:
            continue
        total = 0
        for p in mod.parameters():
            total += p.numel()
            tracked_ids.add(id(p))
        by_mod[name] = total

    # Orphan Parameters directly on snn (ts_weights, *_cdecay, *_vdecay, ...)
    orphan = 0
    for p in snn.parameters():
        if id(p) not in tracked_ids:
            orphan += p.numel()
    by_mod["neuron_state"] = orphan

    return by_mod


# ------------------------------------------------------------------
# FLOP count (structural, per forward call)
# ------------------------------------------------------------------
def count_flops_per_sample(spike_ts: int = SPIKE_TS) -> Dict[str, Dict[str, int]]:
    """Return dense MACs per sample for each top-level submodule.

    Convention: each timestep the full conv / fc / cbam graph is executed.
    fc2 runs once per sample at the end (on the weighted spike sum).
    """
    c1_out = SHAPES["after_conv1"]   # 64,8,9
    c2_out = SHAPES["after_conv2"]   # 128,6,7
    c3_out = SHAPES["after_conv3"]   # 256,1,1
    pool_out = SHAPES["after_pool"]  # 128,3,3

    macs_per_ts = {}

    # Conv blocks
    macs_per_ts["conv1"] = conv2d_macs(1, 64, 3, 3, c1_out[1], c1_out[2])
    macs_per_ts["conv2"] = conv2d_macs(64, 128, 3, 3, c2_out[1], c2_out[2])
    macs_per_ts["conv3"] = conv2d_macs(128, 256, 3, 3, c3_out[1], c3_out[2])

    # CBAM blocks (operate on the post-LIF spike tensor of each conv)
    macs_per_ts["cbam_conv1"] = cbam_macs(64,  c1_out[1], c1_out[2])["macs_total"]
    macs_per_ts["cbam_conv2"] = cbam_macs(128, c2_out[1], c2_out[2])["macs_total"]
    macs_per_ts["cbam_conv3"] = cbam_macs(256, c3_out[1], c3_out[2])["macs_total"]

    # avg_pool: 2x2 avg, no MACs (just additions); treat as 0 MAC.
    macs_per_ts["avg_pool"] = 0

    # temp_conv1: TemporalConvCUBALIFCell with 3 Linear(256,256), rotates
    # through kernel_size=3 membranes (one Linear executed per timestep).
    macs_per_ts["temp_conv1"] = linear_macs(256, 256)

    # fc1: Linear(256,256) every timestep
    macs_per_ts["fc1"] = linear_macs(256, 256)

    # fc2: Linear(256,2), ONCE per sample, not per timestep
    macs_once = {"fc2": linear_macs(256, 2)}

    # Assemble per-sample totals
    per_module = {}
    for k, v in macs_per_ts.items():
        per_module[k] = {
            "macs_per_timestep": v,
            "macs_per_sample": v * spike_ts,
        }
    per_module["fc2"] = {
        "macs_per_timestep": 0,
        "macs_per_sample": macs_once["fc2"],
    }

    return per_module


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    # Instantiating the model requires CUDA (device is hard-coded in snn_n_r_dfbsa).
    # For pure static analysis we can still count params if CUDA is present. If
    # CUDA is missing we fall back to a structural-only path.
    can_build = torch.cuda.is_available()

    if can_build:
        device = torch.device("cuda")
        net = WrapCUBASpikingCNN(SPIKE_TS, device, PARAM_LIST, record_neuron=None)
        net = net.to(device)
        params_by_mod = count_params_by_module(net)
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    else:
        print("[WARN] CUDA not available. Skipping param count (model "
              "instantiation needs CUDA). FLOP counts are still reported.")
        params_by_mod = {}
        total_params = 0
        trainable_params = 0

    flops_by_mod = count_flops_per_sample(SPIKE_TS)

    # ---- aggregate totals ----
    total_macs_per_sample = sum(v["macs_per_sample"] for v in flops_by_mod.values())

    cbam_modules = ["cbam_conv1", "cbam_conv2", "cbam_conv3"]
    cbam_params = sum(params_by_mod.get(m, 0) for m in cbam_modules)
    cbam_macs_sample = sum(flops_by_mod[m]["macs_per_sample"] for m in cbam_modules)
    non_cbam_params = total_params - cbam_params
    non_cbam_macs_sample = total_macs_per_sample - cbam_macs_sample

    # ---- print report ----
    print("=" * 70)
    print("Params by module")
    print("=" * 70)
    print(f"{'module':<16} {'params':>14}")
    for m, p in params_by_mod.items():
        print(f"{m:<16} {p:>14,}")
    print(f"{'TOTAL':<16} {total_params:>14,}")
    print(f"{'  (trainable)':<16} {trainable_params:>14,}")
    print()

    print("=" * 70)
    print(f"MACs per sample (spike_ts={SPIKE_TS}, dense upper bound)")
    print("=" * 70)
    print(f"{'module':<16} {'MACs/timestep':>16} {'MACs/sample':>16} {'%':>7}")
    order = ["conv1", "cbam_conv1", "conv2", "cbam_conv2", "avg_pool",
             "conv3", "cbam_conv3", "temp_conv1", "fc1", "fc2"]
    for m in order:
        if m in flops_by_mod:
            v = flops_by_mod[m]
            pct = (v["macs_per_sample"] / total_macs_per_sample * 100.0
                   if total_macs_per_sample > 0 else 0.0)
            print(f"{m:<16} {v['macs_per_timestep']:>16,} "
                  f"{v['macs_per_sample']:>16,} {pct:>6.2f}%")
    print(f"{'TOTAL':<16} {'':>16} {total_macs_per_sample:>16,} {100.0:>6.2f}%")
    print()

    print("=" * 70)
    print("CBAM attention cost (Q1 headline numbers)")
    print("=" * 70)
    if total_params > 0:
        print(f"CBAM params       : {cbam_params:>12,}  "
              f"({cbam_params/total_params*100:.2f}% of total)")
        print(f"non-CBAM params   : {non_cbam_params:>12,}")
    print(f"CBAM MACs/sample  : {cbam_macs_sample:>12,}  "
          f"({cbam_macs_sample/total_macs_per_sample*100:.2f}% of total)")
    print(f"non-CBAM MACs     : {non_cbam_macs_sample:>12,}")
    print()

    # ---- write artifacts ----
    # CSV: one row per module
    csv_path = OUT_DIR / "params_flops_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "params", "macs_per_timestep", "macs_per_sample",
                    "macs_percent_of_total"])
        for m in order:
            if m not in flops_by_mod:
                continue
            p = params_by_mod.get(m, 0)
            v = flops_by_mod[m]
            pct = (v["macs_per_sample"] / total_macs_per_sample * 100.0
                   if total_macs_per_sample > 0 else 0.0)
            w.writerow([m, p, v["macs_per_timestep"], v["macs_per_sample"],
                        f"{pct:.4f}"])
        w.writerow(["neuron_state", params_by_mod.get("neuron_state", 0), 0, 0, 0])
        w.writerow(["__TOTAL__", total_params, "", total_macs_per_sample, 100.0])
        w.writerow(["__CBAM_AGG__", cbam_params, "", cbam_macs_sample,
                    f"{cbam_macs_sample/total_macs_per_sample*100:.4f}"
                    if total_macs_per_sample > 0 else 0.0])
        w.writerow(["__NON_CBAM_AGG__", non_cbam_params, "", non_cbam_macs_sample,
                    f"{non_cbam_macs_sample/total_macs_per_sample*100:.4f}"
                    if total_macs_per_sample > 0 else 0.0])

    # Markdown summary for the paper
    md_path = OUT_DIR / "params_flops_summary.md"
    with open(md_path, "w") as f:
        f.write("# Params / FLOPs — WrapCUBASpikingCNN (snn_n_r_dfbsa.py)\n\n")
        f.write(f"Input shape per timestep: `[B, 1, 10, 11]`  ·  spike_ts = {SPIKE_TS}\n\n")
        f.write("FLOPs here = dense MACs (upper bound). Real SNN cost equals "
                "MACs × mean spike rate; that number is reported separately by "
                "`synops_energy.py` once a checkpoint is available.\n\n")

        f.write("## Per-module table\n\n")
        f.write("| module | params | MACs / timestep | MACs / sample | % of total |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for m in order:
            if m not in flops_by_mod:
                continue
            p = params_by_mod.get(m, 0)
            v = flops_by_mod[m]
            pct = (v["macs_per_sample"] / total_macs_per_sample * 100.0
                   if total_macs_per_sample > 0 else 0.0)
            f.write(f"| `{m}` | {p:,} | {v['macs_per_timestep']:,} | "
                    f"{v['macs_per_sample']:,} | {pct:.2f}% |\n")
        f.write(f"| neuron_state (ts_weights, cdecay/vdecay) | "
                f"{params_by_mod.get('neuron_state', 0):,} | 0 | 0 | 0% |\n")
        f.write(f"| **total** | **{total_params:,}** | — | "
                f"**{total_macs_per_sample:,}** | **100%** |\n\n")

        f.write("## CBAM aggregate (attention cost)\n\n")
        if total_params > 0:
            f.write(f"- CBAM params: **{cbam_params:,}** "
                    f"({cbam_params/total_params*100:.2f}% of total)\n")
            f.write(f"- non-CBAM params: {non_cbam_params:,}\n")
        f.write(f"- CBAM MACs/sample: **{cbam_macs_sample:,}** "
                f"({cbam_macs_sample/total_macs_per_sample*100:.2f}% of total)\n")
        f.write(f"- non-CBAM MACs/sample: {non_cbam_macs_sample:,}\n")

    # JSON dump for machine consumption (e.g. synops_energy.py reuse)
    json_path = OUT_DIR / "params_flops.json"
    payload = {
        "config": {"spike_ts": SPIKE_TS, "param_list": PARAM_LIST},
        "params_by_module": params_by_mod,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops_by_module": flops_by_mod,
        "total_macs_per_sample": total_macs_per_sample,
        "cbam_params": cbam_params,
        "cbam_macs_per_sample": cbam_macs_sample,
        "non_cbam_params": non_cbam_params,
        "non_cbam_macs_per_sample": non_cbam_macs_sample,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"Artifacts written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
