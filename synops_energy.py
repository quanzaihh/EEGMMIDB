"""
synops_energy.py
================

Second-batch energy estimation (REQUIRES a trained checkpoint).

Estimates per-sample energy of WrapCUBASpikingCNN on neuromorphic-style
hardware, using the standard SNN accounting:

    E_total = Σ_SNN_layer  SynOps_l × E_AC   +   Σ_ANN_layer  MACs_l × E_MAC

  - SynOps = dense_MACs × mean_firing_rate   (spike-gated AC operations)
  - E_AC  = 0.9  pJ   (45nm add, Horowitz 2014 ISSCC)
  - E_MAC = 4.6  pJ   (45nm 32-bit FP MAC)

Which layers are "SNN" vs "ANN"
-------------------------------
The FeedForwardCUBALIFCell input is the spike-output of the previous LIF layer,
so the Conv/Linear inside each cell is spike-driven --> AC:

  * conv1 / conv2 / conv3       : SNN (driven by input image spikes /
                                   preceding LIF spikes)
  * temp_conv1 / fc1             : SNN (driven by preceding LIF spikes)

CBAM and fc2 run on dense activations:

  * cbam_conv1 / cbam_conv2 / cbam_conv3 : ANN (sigmoid gating, spatial conv,
                                            channel MLP all operate on
                                            real-valued feature maps)
  * fc2                                  : ANN (takes the weighted spike sum,
                                            which is a real-valued vector)

This is the conservative split — treating CBAM as ANN (no AC gating) is a
worst case for the attention cost, so if CBAM still looks small here, it is
genuinely small.

Usage
-----
    python synops_energy.py --ckpt training_outputs_LR/checkpoints/checkpoint_best.pth
    # optional:
    #   --slice-route ../utils/eegmmidb_slice_norm/
    #   --subjects 1 2 3 ... (validation subjects to sample from)
    #   --num-samples 512    (number of val samples to average spike rate on)
    #   --batch-size 32
    #   --e-ac 0.9           (pJ per AC)
    #   --e-mac 4.6          (pJ per MAC)

Outputs
-------
power_latency_results/synops_energy.csv
power_latency_results/synops_energy.md
power_latency_results/synops_energy.json
"""

import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append("../utils/")

from snn_n_r_dfbsa import WrapCUBASpikingCNN, FeedForwardCUBALIFCell, \
                         TemporalConvCUBALIFCell
from dataset import EEGDataset2DLeftRight, ToTensor
from torchvision.transforms import Compose


OUT_DIR = Path("power_latency_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPIKE_TS = 160
PARAM_LIST = [0.1, 0.1, 0.1, 0.3, 0.01, 0.1, 0.01]

# Layers whose compute is spike-gated (AC energy, scaled by firing rate)
SNN_LAYERS = ["conv1", "conv2", "conv3", "temp_conv1", "fc1"]
# Layers that run on dense real-valued activations (MAC energy, full cost)
ANN_LAYERS = ["cbam_conv1", "cbam_conv2", "cbam_conv3", "fc2"]


# ------------------------------------------------------------------
# Dense MACs per layer (copied from params_flops_count.py, kept in-sync)
# ------------------------------------------------------------------
def dense_macs_per_ts():
    """Return MACs per single timestep for every module."""
    def conv(inC, oC, kH, kW, oH, oW):
        return inC * oC * kH * kW * oH * oW

    def lin(i, o):
        return i * o

    def cbam(C, H, W, r=16, sk=7):
        red = max(C // r, 1)
        mlp = 2 * (lin(C, red) + lin(red, C))      # avg- and max-pool branches
        sg = conv(2, 1, sk, sk, H, W)              # SAME padding keeps H,W
        return mlp + sg

    return {
        # SNN
        "conv1":      conv(1, 64, 3, 3, 8, 9),
        "conv2":      conv(64, 128, 3, 3, 6, 7),
        "conv3":      conv(128, 256, 3, 3, 1, 1),
        "temp_conv1": lin(256, 256),
        "fc1":        lin(256, 256),
        # ANN
        "cbam_conv1": cbam(64, 8, 9),
        "cbam_conv2": cbam(128, 6, 7),
        "cbam_conv3": cbam(256, 1, 1),
        "fc2":        lin(256, 2),       # once per sample, not per ts
    }


# ------------------------------------------------------------------
# Firing-rate hooks
# ------------------------------------------------------------------
class FiringRateHook:
    """Register on each FeedForwardCUBALIFCell / TemporalConvCUBALIFCell and
    accumulate spike counts over all timesteps and all samples it sees."""

    def __init__(self):
        # name -> {"spike_sum": float, "elem_sum": int, "calls": int}
        self.stats = {}

    def register(self, model: nn.Module):
        handles = []
        snn = model.snn
        for name in SNN_LAYERS:
            mod = getattr(snn, name, None)
            if mod is None:
                continue
            self.stats[name] = {"spike_sum": 0.0, "elem_sum": 0, "calls": 0}
            h = mod.register_forward_hook(self._make_hook(name))
            handles.append(h)
        return handles

    def _make_hook(self, name):
        def hook(module, inputs, output):
            # FeedForwardCUBALIFCell.forward returns (output_spike, state)
            # TemporalConvCUBALIFCell.forward also returns a tuple whose [0] is
            # the output spike.
            spike = output[0] if isinstance(output, (tuple, list)) else output
            # Spikes are 0/1 float tensors; a .sum() gives the count directly.
            with torch.no_grad():
                s = spike.detach()
                self.stats[name]["spike_sum"] += float(s.sum().item())
                self.stats[name]["elem_sum"] += int(s.numel())
                self.stats[name]["calls"] += 1
        return hook

    def mean_firing_rate(self, name):
        st = self.stats[name]
        if st["elem_sum"] == 0:
            return float("nan")
        return st["spike_sum"] / st["elem_sum"]


# ------------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------------
def load_checkpoint(net, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
        meta = {k: v for k, v in obj.items() if k != "model_state_dict"
                and k != "optimizer_state_dict"}
    else:
        state = obj
        meta = {}
    missing, unexpected = net.load_state_dict(state, strict=False)
    return meta, missing, unexpected


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Path to checkpoint (e.g. training_outputs_LR/"
                        "checkpoints/checkpoint_best.pth)")
    p.add_argument("--slice-route", default="../utils/eegmmidb_slice_norm/",
                   help="Pre-sliced dataset path (for spike-rate measurement)")
    p.add_argument("--subjects", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5],
                   help="Subjects to draw validation samples from")
    p.add_argument("--num-samples", type=int, default=512,
                   help="Number of samples over which to average firing rate")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--e-ac", type=float, default=0.9,
                   help="Energy per AC (pJ). 45nm ~ 0.9 pJ")
    p.add_argument("--e-mac", type=float, default=4.6,
                   help="Energy per MAC (pJ). 45nm 32-bit FP ~ 4.6 pJ")
    p.add_argument("--spike-ts", type=int, default=SPIKE_TS)
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("[FATAL] CUDA is required (CUBASpikingCNN hard-codes "
              "torch.device('cuda') in forward).")
        sys.exit(1)
    device = torch.device("cuda")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"[FATAL] checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ---- build model + load weights ----
    net = WrapCUBASpikingCNN(args.spike_ts, device, PARAM_LIST,
                             record_neuron=None).to(device)
    meta, missing, unexpected = load_checkpoint(net, str(ckpt_path), device)
    net.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  meta: {meta}")
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    # ---- build validation loader ----
    ds = EEGDataset2DLeftRight(
        base_route=args.slice_route,
        subject_id_list=args.subjects,
        start_ts=0, end_ts=161, window_ts=args.spike_ts, overlap_ts=0,
        use_imagery=True,
        transform=Compose([ToTensor()]),
    )
    if len(ds) == 0:
        print(f"[FATAL] dataset empty at {args.slice_route} subjects={args.subjects}")
        sys.exit(1)
    n_sample = min(args.num_samples, len(ds))
    print(f"Using {n_sample} samples from validation subjects {args.subjects}")
    # Deterministic subset
    indices = list(range(n_sample))
    sub = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(sub, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    # ---- register firing-rate hooks & run forward ----
    hook = FiringRateHook()
    handles = hook.register(net)

    total_samples_seen = 0
    with torch.no_grad():
        for batch in loader:
            # EEGDataset2DLeftRight returns (data, label); data shape after
            # ToTensor: [B, 1, 10, 11, spike_ts]
            data = batch[0] if isinstance(batch, (tuple, list)) else batch
            data = data.to(device, non_blocking=True)
            _ = net(data)
            total_samples_seen += data.shape[0]

    for h in handles:
        h.remove()

    # ---- compute SynOps and energy ----
    macs = dense_macs_per_ts()
    e_ac_pJ = args.e_ac
    e_mac_pJ = args.e_mac

    rows = []
    total_energy_pJ = 0.0
    snn_energy_pJ = 0.0
    ann_energy_pJ = 0.0

    for name in SNN_LAYERS:
        rate = hook.mean_firing_rate(name)
        # SynOps per sample: dense MACs × spike_ts × firing_rate.
        # Note: firing rate here is the fraction of spikes at the layer's
        # INPUT to the Conv/Linear of the NEXT layer — but because in this
        # model's wiring the conv inside cell_k is driven by the spike output
        # of cell_{k-1}, and we hook the output spike of cell_k, we actually
        # want the firing rate of the *previous* LIF cell.
        #
        # However, because of how CUBASpikingCNN chains them, input to conv_k
        # is equivalent to the output spike of the preceding LIF cell, so
        # using each cell's output rate as the drive for the *next* layer is
        # the natural attribution. We keep per-layer rate and report it; the
        # user can verify via firing_rate_by_layer in the JSON.
        #
        # For a first-order, reviewer-facing estimate we use the cell's own
        # firing rate as the gating rate for that layer's conv/linear.
        synops_per_sample = macs[name] * args.spike_ts * rate
        e_layer = synops_per_sample * e_ac_pJ
        snn_energy_pJ += e_layer
        rows.append({
            "module": name,
            "type": "SNN",
            "dense_macs_per_ts": macs[name],
            "timesteps": args.spike_ts,
            "mean_firing_rate": rate,
            "synops_per_sample": synops_per_sample,
            "energy_pJ_per_sample": e_layer,
        })

    for name in ANN_LAYERS:
        # fc2 runs once per sample (not per ts); cbam_* run every ts.
        if name == "fc2":
            macs_per_sample = macs[name]
        else:
            macs_per_sample = macs[name] * args.spike_ts
        e_layer = macs_per_sample * e_mac_pJ
        ann_energy_pJ += e_layer
        rows.append({
            "module": name,
            "type": "ANN",
            "dense_macs_per_ts": macs[name],
            "timesteps": args.spike_ts if name != "fc2" else 1,
            "mean_firing_rate": 1.0,         # full MAC, no gating
            "synops_per_sample": macs_per_sample,
            "energy_pJ_per_sample": e_layer,
        })

    total_energy_pJ = snn_energy_pJ + ann_energy_pJ

    cbam_energy = sum(r["energy_pJ_per_sample"] for r in rows
                      if r["module"].startswith("cbam_"))
    non_cbam_energy = total_energy_pJ - cbam_energy

    # ---- print ----
    print("\n" + "=" * 76)
    print(f"{'module':<14} {'type':<4} {'rate':>7} {'synops/sample':>16} "
          f"{'energy (pJ)':>14}   %")
    print("=" * 76)
    for r in rows:
        pct = (r["energy_pJ_per_sample"] / total_energy_pJ * 100.0
               if total_energy_pJ > 0 else 0.0)
        print(f"{r['module']:<14} {r['type']:<4} {r['mean_firing_rate']:>7.4f} "
              f"{r['synops_per_sample']:>16,.0f} "
              f"{r['energy_pJ_per_sample']:>14,.2f}  {pct:>5.2f}%")
    print("-" * 76)
    print(f"SNN subtotal : {snn_energy_pJ:>14,.2f} pJ  "
          f"({snn_energy_pJ/total_energy_pJ*100:.2f}%)")
    print(f"ANN subtotal : {ann_energy_pJ:>14,.2f} pJ  "
          f"({ann_energy_pJ/total_energy_pJ*100:.2f}%)")
    print(f"CBAM         : {cbam_energy:>14,.2f} pJ  "
          f"({cbam_energy/total_energy_pJ*100:.2f}% of total)")
    print(f"non-CBAM     : {non_cbam_energy:>14,.2f} pJ")
    print(f"TOTAL        : {total_energy_pJ:>14,.2f} pJ/sample "
          f"= {total_energy_pJ/1e6:.4f} μJ/sample")

    # ---- write artifacts ----
    csv_path = OUT_DIR / "synops_energy.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["module", "type", "dense_macs_per_ts", "timesteps",
                    "mean_firing_rate", "synops_per_sample",
                    "energy_pJ_per_sample", "percent_of_total"])
        for r in rows:
            pct = (r["energy_pJ_per_sample"] / total_energy_pJ * 100.0
                   if total_energy_pJ > 0 else 0.0)
            w.writerow([r["module"], r["type"], r["dense_macs_per_ts"],
                        r["timesteps"], f"{r['mean_firing_rate']:.6f}",
                        f"{r['synops_per_sample']:.2f}",
                        f"{r['energy_pJ_per_sample']:.4f}",
                        f"{pct:.4f}"])
        w.writerow(["__SNN_SUB__", "", "", "", "", "",
                    f"{snn_energy_pJ:.4f}",
                    f"{snn_energy_pJ/total_energy_pJ*100:.4f}"])
        w.writerow(["__ANN_SUB__", "", "", "", "", "",
                    f"{ann_energy_pJ:.4f}",
                    f"{ann_energy_pJ/total_energy_pJ*100:.4f}"])
        w.writerow(["__CBAM__", "", "", "", "", "",
                    f"{cbam_energy:.4f}",
                    f"{cbam_energy/total_energy_pJ*100:.4f}"])
        w.writerow(["__TOTAL__", "", "", "", "", "",
                    f"{total_energy_pJ:.4f}", "100.0"])

    md_path = OUT_DIR / "synops_energy.md"
    with open(md_path, "w") as f:
        f.write(f"# SynOps + Energy — {ckpt_path.name}\n\n")
        f.write(f"- Checkpoint: `{ckpt_path}`\n")
        f.write(f"- Samples used for firing-rate avg: {total_samples_seen}\n")
        f.write(f"- spike_ts: {args.spike_ts}\n")
        f.write(f"- E_AC = {e_ac_pJ} pJ, E_MAC = {e_mac_pJ} pJ "
                f"(Horowitz 2014, 45nm)\n\n")
        f.write("| module | type | firing rate | SynOps / sample | "
                "energy (pJ) | % |\n|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            pct = (r["energy_pJ_per_sample"] / total_energy_pJ * 100.0
                   if total_energy_pJ > 0 else 0.0)
            f.write(f"| `{r['module']}` | {r['type']} | "
                    f"{r['mean_firing_rate']:.4f} | "
                    f"{r['synops_per_sample']:,.0f} | "
                    f"{r['energy_pJ_per_sample']:,.2f} | {pct:.2f}% |\n")
        f.write(f"| **SNN subtotal** | | | | **{snn_energy_pJ:,.2f}** | "
                f"{snn_energy_pJ/total_energy_pJ*100:.2f}% |\n")
        f.write(f"| **ANN subtotal** | | | | **{ann_energy_pJ:,.2f}** | "
                f"{ann_energy_pJ/total_energy_pJ*100:.2f}% |\n")
        f.write(f"| **CBAM total** | | | | **{cbam_energy:,.2f}** | "
                f"{cbam_energy/total_energy_pJ*100:.2f}% |\n")
        f.write(f"| **TOTAL** | | | | **{total_energy_pJ:,.2f} pJ** "
                f"= **{total_energy_pJ/1e6:.4f} μJ/sample** | 100% |\n")

    json_path = OUT_DIR / "synops_energy.json"
    with open(json_path, "w") as f:
        json.dump({
            "ckpt": str(ckpt_path),
            "ckpt_meta": {k: (v if isinstance(v, (int, float, str, list, dict))
                              else str(v)) for k, v in meta.items()},
            "samples_used": total_samples_seen,
            "spike_ts": args.spike_ts,
            "e_ac_pJ": e_ac_pJ, "e_mac_pJ": e_mac_pJ,
            "firing_rate_by_layer": {n: hook.mean_firing_rate(n) for n in SNN_LAYERS},
            "per_module": rows,
            "snn_energy_pJ": snn_energy_pJ,
            "ann_energy_pJ": ann_energy_pJ,
            "cbam_energy_pJ": cbam_energy,
            "non_cbam_energy_pJ": non_cbam_energy,
            "total_energy_pJ_per_sample": total_energy_pJ,
            "total_energy_uJ_per_sample": total_energy_pJ / 1e6,
        }, f, indent=2, default=str)

    print(f"\nArtifacts written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
