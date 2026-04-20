"""
TemporalConv T 值对比实验（CUDA 服务器版）
==========================================

相对 CPU 版本 (run_T_experiments.py) 的差异：
  1. 启动时强校验 CUDA 可用，否则直接退出（避免误跑到 CPU）
  2. 支持多卡 DataParallel（CUDA_VISIBLE_DEVICES 控制用哪几张）
  3. 开启 pin_memory + num_workers=4 提升数据加载吞吐
  4. 开启 cudnn.benchmark 让卷积核自动挑最快实现
  5. 支持通过命令行 / 环境变量覆盖 EPOCHS / BATCH_SIZE / T_VALUES，方便在服务器上做消融

依赖文件（需要和本脚本放在同一目录）：
  - snn_T_experiment.py      (device-agnostic，用 self.device)
  - dataset.py, utility.py, cbam*.py
  - eegmmidb_slice_norm/     (数据集)

使用示例：
  # 单卡全量：
  python run_T_experiments_cuda.py

  # 指定 GPU 2 号：
  CUDA_VISIBLE_DEVICES=2 python run_T_experiments_cuda.py

  # 多卡 DataParallel：
  CUDA_VISIBLE_DEVICES=0,1,2,3 python run_T_experiments_cuda.py

  # 缩水扫描（快速跑）：
  python run_T_experiments_cuda.py --t-values 1 3 10 --epochs 20 --batch-size 128
"""

import sys
import os
sys.path.append("../utils/")

import argparse
import json
import time
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from snn_T_experiment import WrapCUBASpikingCNN
from dataset import (ToTensor, EEGDataset2DLeftRight,
                     EEGDatasetLeftFeet, EEGDatasetRightFeet)
from utility import train_validate_split_subjects, samples_per_class


# 任务名 -> Dataset 类（CLI 和目录名用 key，内部用 value）
DATASET_REGISTRY = {
    "LeftRight": EEGDataset2DLeftRight,
    "LeftFeet":  EEGDatasetLeftFeet,
    "RightFeet": EEGDatasetRightFeet,
}
DEFAULT_DATASETS = ["LeftRight", "LeftFeet", "RightFeet"]


# ============================================================
# 默认实验配置（可被命令行参数覆盖）
# ============================================================
DEFAULT_T_VALUES = [1, 2, 3, 5, 10]
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
SPIKE_TS = 160              # 固定
SEED = 4

# 学习率
WT_LR = 0.0001
TS_LR = 0.0001
NEURON_LR = 0.0001

# 神经元参数
PARAM_LIST = [0.1, 0.1, 0.1, 0.3, 0.01, 0.1, 0.01]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ds_params = {
    "base_route": os.path.join(BASE_DIR, "eegmmidb_slice_norm/"),
    "subject_id_list": [i + 1 for i in range(109) if not (i + 1 in [88, 89, 92, 100, 104, 106])],
    "start_ts": 0,
    "end_ts": 161,
    "window_ts": 160,
    "overlap_ts": 0,
    "use_imagery": False,
    "transform": ToTensor()
}

# 10 折交叉验证分组（默认使用第 0 折做验证集）
val_list = [
    [i + 1 for i in range(10)],
    [i + 11 for i in range(10)],
    [i + 21 for i in range(10)],
    [i + 31 for i in range(10)],
    [i + 41 for i in range(10)],
    [i + 51 for i in range(10)],
    [i + 61 for i in range(10)],
    [i + 71 for i in range(10)],
    [81, 82, 83, 84, 85, 86, 87, 90, 91, 93],
    [94, 95, 96, 97, 98, 99, 101, 102, 103, 105]
]

RESULTS_DIR = Path("results_T_experiment_cuda")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 辅助函数
# ============================================================
def check_cuda_or_die():
    """强校验 CUDA；没有则直接退出，避免误跑 CPU"""
    if not torch.cuda.is_available():
        print("[FATAL] CUDA 不可用。本脚本是 CUDA 专用版。")
        print("         如要在 CPU/MPS 上跑，请使用 run_T_experiments.py。")
        sys.exit(1)
    n = torch.cuda.device_count()
    print(f"[INFO] CUDA OK, 可见 GPU 数: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"       [{i}] {props.name}, {props.total_memory / 1024**3:.1f} GB")
    print(f"[INFO] PyTorch: {torch.__version__}, CUDA runtime: {torch.version.cuda}")


def setup_logger(T, dataset_name="LeftRight"):
    """为每个 (数据集, T 值) 组合设置独立日志目录和 logger"""
    log_dir = RESULTS_DIR / dataset_name / f"T{T}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f'{dataset_name}_T_{T}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_dir / "train.log", encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger, log_dir


def set_seed(seed: int):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # 确定性优先于性能


def test_accuracy(network, test_loader, device):
    network.eval()
    with torch.no_grad():
        class_correct = None
        class_total = None
        all_correct = 0
        all_total = 0
        for data in test_loader:
            eeg_data, label = data
            eeg_data = eeg_data.to(device, non_blocking=True)
            output = network(eeg_data)
            _, predicted = torch.max(output, 1)
            pred_np = predicted.cpu().numpy()
            label_np = label.cpu().numpy()
            if class_correct is None:
                n_classes = int(label_np.max()) + 1
                class_correct = np.zeros(n_classes, dtype=np.int64)
                class_total = np.zeros(n_classes, dtype=np.int64)
            eq = np.equal(pred_np, label_np)
            for i in range(label_np.shape[0]):
                la = int(label_np[i])
                class_total[la] += 1
                if bool(eq[i]):
                    class_correct[la] += 1
                    all_correct += 1
                all_total += 1
    overall = float(all_correct) / float(all_total) if all_total > 0 else 0.0
    class_acc = (class_correct.astype(float) / (class_total + 1e-8)) if class_total is not None else np.zeros(1)
    network.train()
    return overall, class_acc


# ============================================================
# 训练函数
# ============================================================
def train_network_with_T(temporal_kernel, dataset_kwargs, spike_ts, param_list,
                         validate_subject_list, batch_size, epoch, lr, logger, log_dir,
                         seed=None, num_workers=4, lr_min=1e-6,
                         dataset_cls=EEGDataset2DLeftRight):
    device = torch.device("cuda")

    if seed is not None:
        set_seed(int(seed))
        logger.info('Set random seed to: %d', int(seed))

    # 模型
    net = WrapCUBASpikingCNN(spike_ts, device, param_list=param_list, temporal_kernel=temporal_kernel)
    net = net.to(device)

    # 多卡 DataParallel
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        logger.info('Using DataParallel across %d GPUs', n_gpus)
        net = nn.DataParallel(net)
    net.train()

    # 参数统计
    base_net = net.module if isinstance(net, nn.DataParallel) else net
    total_params, trainable_params = base_net.count_parameters()
    tc_params = base_net.count_temporal_conv_parameters()
    logger.info(f'Temporal Kernel T = {temporal_kernel}')
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    logger.info(f'TemporalConv parameters: {tc_params:,}')

    # 非负权重约束：对「进最后一个 temporal_conv 前一层」(conv3 的 Conv2d) 的权重
    # 做投影梯度下降——训练开始前投影一次，每个 optimizer.step() 后再 clamp 一次。
    # 路径: base_net.snn.conv3.psp_func = nn.Conv2d(128, 256, (3,3), bias=True)
    pre_tc_layer = base_net.snn.conv3.psp_func
    with torch.no_grad():
        pre_tc_layer.weight.data.clamp_(min=0.0)
    logger.info('Pre-TC layer (snn.conv3.psp_func, Conv2d) weight constrained to non-negative, shape=%s',
                tuple(pre_tc_layer.weight.shape))

    # 数据集（通过 dataset_cls 参数化，支持 LeftRight / LeftFeet / RightFeet）
    train_ds_kwargs = dataset_kwargs.copy()
    train_ds_kwargs.setdefault("transform", ToTensor())
    train_ds = dataset_cls(**train_ds_kwargs)

    val_ds_kwargs = dataset_kwargs.copy()
    val_ds_kwargs.setdefault("transform", ToTensor())
    val_ds = dataset_cls(**val_ds_kwargs)
    logger.info(f"Dataset class: {dataset_cls.__name__}")

    train_indices, val_indices = train_validate_split_subjects(train_ds, validate_subject_list)

    logger.info("Training Samples per Class:")
    samples_per_class(train_ds.label[train_indices])
    logger.info("Validate Samples per Class:")
    samples_per_class(val_ds.label[val_indices])

    train_sampler = sampler.SubsetRandomSampler(train_indices)
    val_sampler = sampler.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              sampler=train_sampler, num_workers=num_workers,
                              pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=num_workers,
                            pin_memory=True, persistent_workers=(num_workers > 0))

    # 优化器：decay / ts_weights / 其它权重分三组学习率
    try:
        decays = ['module.snn.c1_vdecay', 'module.snn.c2_vdecay', 'module.snn.c3_vdecay',
                  'module.snn.tc1_vdecay', 'module.snn.tc1_cdecay', 'module.snn.r1_vdecay',
                  'module.snn.f1_vdecay', 'module.snn.c1_cdecay', 'module.snn.c2_cdecay',
                  'module.snn.c3_cdecay', 'module.snn.r1_cdecay', 'module.snn.f1_cdecay']
        ts_weights = ['module.snn.ts_weights']
        # DataParallel 之外的场景名字没有 module. 前缀，两种都试
        if not isinstance(net, nn.DataParallel):
            decays = [d.replace('module.', '') for d in decays]
            ts_weights = [d.replace('module.', '') for d in ts_weights]
        decay_params = [p for n, p in net.named_parameters() if n in decays]
        ts_params = [p for n, p in net.named_parameters() if n in ts_weights]
        other = [p for n, p in net.named_parameters() if n not in decays + ts_weights]
        optimizer = optim.Adam([{'params': other},
                                {'params': decay_params, 'lr': lr[0]},
                                {'params': ts_params, 'lr': lr[1]}], lr=lr[2])
    except Exception as e:
        logger.warning("param-group split failed (%s), fallback to single lr", e)
        optimizer = optim.Adam(net.parameters(), lr=lr[2] if len(lr) > 2 else 1e-4)

    criterion = nn.CrossEntropyLoss()
    # 学习率退火：CosineAnnealingLR，每个 epoch 末 step() 一次，
    # T_max = 总 epoch 数，eta_min 为最小学习率下限
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=lr_min)
    logger.info('LR scheduler: CosineAnnealingLR(T_max=%d, eta_min=%g)', epoch, lr_min)

    # 训练循环
    epoch_accs, epoch_losses, epoch_class_accs = [], [], []
    test_best_acc = 0.0
    best_epoch = 0

    for e in range(epoch):
        epoch_start = time.time()
        running_loss = 0.0
        iters = 0
        for data in train_loader:
            eeg_data, label = data
            eeg_data = eeg_data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = net(eeg_data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # 投影梯度下降：把 conv3(进 temp_conv 前一层) 的权重投影回非负锥
            with torch.no_grad():
                pre_tc_layer.weight.data.clamp_(min=0.0)

            running_loss += loss.item()
            iters += 1

        net.eval()
        acc, class_acc = test_accuracy(net, val_loader, device)
        net.train()

        if acc > test_best_acc:
            test_best_acc = acc
            best_epoch = e
            ckpt_dir = log_dir / 'checkpoints'
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model_state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            torch.save({
                'epoch': int(e),
                'temporal_kernel': temporal_kernel,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
            }, str(ckpt_dir / "checkpoint_best.pth"))

        avg_loss = running_loss / max(1, iters)
        epoch_accs.append(float(acc))
        epoch_losses.append(float(avg_loss))
        epoch_class_accs.append(class_acc.copy())

        epoch_time = time.time() - epoch_start
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        # 监控 conv3(进 temp_conv 前一层) 权重的 min/max，确认非负约束始终成立
        with torch.no_grad():
            w = pre_tc_layer.weight.data
            w_min = float(w.min().item())
            w_max = float(w.max().item())
            w_neg = int((w < 0).sum().item())
        logger.info('Epoch: %d, Loss: %.6f, Val Acc: %.3f %%, LR: %s, Time: %.1fs, '
                    'GPU mem: %.1f GB, conv3.weight min/max: %.4e/%.4e, neg#: %d',
                    e, avg_loss, acc * 100,
                    ', '.join(f'{x:.2e}' for x in current_lrs),
                    epoch_time,
                    torch.cuda.max_memory_allocated() / 1024**3,
                    w_min, w_max, w_neg)
        assert w_min >= 0.0, f'conv3.psp_func.weight has negative entry: {w_min}'
        torch.cuda.reset_peak_memory_stats()

        # CosineAnnealingLR 按 epoch 退火，不依赖验证指标
        scheduler.step()

    # 保存训练曲线
    plt.figure(figsize=(10, 5))
    epochs_arr = np.arange(1, len(epoch_accs) + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr, epoch_accs, marker='o')
    plt.title(f'Validation Accuracy (T={temporal_kernel})')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_arr, epoch_losses, marker='x', color='red')
    plt.title(f'Training Loss (T={temporal_kernel})')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(log_dir / f'training_curve_T{temporal_kernel}.png', dpi=150)
    plt.close()

    metrics = {
        'epoch_accs': epoch_accs,
        'epoch_losses': epoch_losses,
        'epoch_class_accs': epoch_class_accs,
        'best_acc': test_best_acc,
        'best_epoch': best_epoch,
        'total_params': total_params,
        'tc_params': tc_params
    }
    return net, metrics


# ============================================================
# 主流程
# ============================================================
def run_experiments(t_values, epochs, batch_size, num_workers, lr_min=1e-6,
                    datasets=None):
    datasets = datasets or DEFAULT_DATASETS
    all_results = {}   # {dataset_name: {T: {...}}}

    print("\n" + "=" * 60)
    print("TemporalConv T 值对比实验 (CUDA)")
    print(f"Datasets: {datasets}")
    print(f"T 值: {t_values}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Spike TS: {SPIKE_TS}")
    print(f"Num workers: {num_workers}")
    print(f"LR schedule: CosineAnnealingLR, eta_min = {lr_min:g}")
    print(f"总训练次数: {len(datasets)} datasets × {len(t_values)} T values = "
          f"{len(datasets) * len(t_values)} 次")
    print("=" * 60 + "\n")

    for ds_name in datasets:
        if ds_name not in DATASET_REGISTRY:
            print(f"[WARN] 未知 dataset '{ds_name}', 已知: {list(DATASET_REGISTRY)}")
            continue
        ds_cls = DATASET_REGISTRY[ds_name]
        all_results[ds_name] = {}

        print("\n" + "#" * 70)
        print(f"# Dataset: {ds_name}  ({ds_cls.__name__})")
        print("#" * 70)

        for T in t_values:
            print(f"\n{'='*60}")
            print(f"开始训练 [{ds_name}] T={T}")
            print("="*60)

            logger, log_dir = setup_logger(T, ds_name)
            logger.info(f"Dataset: {ds_name} ({ds_cls.__name__})")

            start_time = time.time()
            model, metrics = train_network_with_T(
                temporal_kernel=T,
                dataset_kwargs=ds_params,
                spike_ts=SPIKE_TS,
                param_list=PARAM_LIST,
                validate_subject_list=val_list[0],
                batch_size=batch_size,
                epoch=epochs,
                lr=[NEURON_LR, TS_LR, WT_LR],
                logger=logger,
                log_dir=log_dir,
                seed=SEED,
                num_workers=num_workers,
                lr_min=lr_min,
                dataset_cls=ds_cls,
            )
            train_time = time.time() - start_time

            all_results[ds_name][T] = {
                'best_val_acc': metrics['best_acc'],
                'best_epoch': metrics['best_epoch'],
                'final_val_acc': metrics['epoch_accs'][-1],
                'train_time_sec': train_time,
                'train_time_min': train_time / 60,
                'total_params': metrics['total_params'],
                'tc_params': metrics['tc_params'],
                'epoch_accs': metrics['epoch_accs'],
                'epoch_losses': metrics['epoch_losses'],
            }

            print(f"\n[{ds_name}] T={T} 完成:")
            print(f"  最佳准确率: {metrics['best_acc']*100:.2f}% (Epoch {metrics['best_epoch']})")
            print(f"  训练时间: {train_time/60:.1f} 分钟")
            print(f"  TemporalConv 参数量: {metrics['tc_params']:,}")

        # 每个数据集各自输出一份 summary + 一张对比图
        ds_tag = "_".join(str(t) for t in t_values)
        ds_summary = RESULTS_DIR / ds_name / f"results_summary_T{ds_tag}.json"
        ds_summary.parent.mkdir(parents=True, exist_ok=True)
        with open(ds_summary, 'w') as f:
            json.dump(all_results[ds_name], f, indent=2)
        plot_comparison(all_results[ds_name], dataset_name=ds_name)

    # 跨数据集总 summary，文件名带 datasets 和 T 列表，避免并行 launcher 覆盖
    ds_key = "_".join(datasets)
    t_key = "_".join(str(t) for t in t_values)
    results_file = RESULTS_DIR / f"results_summary_{ds_key}_T{t_key}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # 打印跨数据集汇总表
    print("\n" + "=" * 78)
    print("跨数据集 × T 值 实验结果汇总")
    print("=" * 78)
    header = f"{'Dataset':<12} {'T':<4} " \
             f"{'Best Acc':<11} {'Best Ep':<8} {'Train(min)':<11} {'TC Params':<12}"
    print(header)
    print("-" * 78)
    for ds_name, res in all_results.items():
        for T, r in res.items():
            print(f"{ds_name:<12} {T:<4} "
                  f"{r['best_val_acc']*100:>6.2f}%    "
                  f"{r['best_epoch']:<8} {r['train_time_min']:>7.1f}     "
                  f"{r['tc_params']:,}")

    print(f"\n结果已保存到: {RESULTS_DIR}")
    return all_results


def plot_comparison(results, dataset_name="LeftRight"):
    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    T_values = list(results.keys())
    best_accs = [results[T]['best_val_acc'] * 100 for T in T_values]
    train_times = [results[T]['train_time_min'] for T in T_values]
    tc_params = [results[T]['tc_params'] for T in T_values]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar([str(t) for t in T_values], best_accs, color='steelblue')
    axes[0].set_xlabel('Temporal Kernel (T)'); axes[0].set_ylabel('Best Validation Accuracy (%)')
    axes[0].set_title(f'[{dataset_name}] Accuracy vs T'); axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(best_accs):
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center')

    axes[1].bar([str(t) for t in T_values], train_times, color='coral')
    axes[1].set_xlabel('Temporal Kernel (T)'); axes[1].set_ylabel('Training Time (min)')
    axes[1].set_title(f'[{dataset_name}] Training Time vs T'); axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(train_times):
        axes[1].text(i, v + 0.5, f'{v:.1f}', ha='center')

    axes[2].bar([str(t) for t in T_values], tc_params, color='green')
    axes[2].set_xlabel('Temporal Kernel (T)'); axes[2].set_ylabel('TemporalConv Parameters')
    axes[2].set_title(f'[{dataset_name}] Parameters vs T'); axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(tc_params):
        axes[2].text(i, v, f'{v//1000}K', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "comparison_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    for T in T_values:
        epochs = range(1, len(results[T]['epoch_accs']) + 1)
        accs = [a * 100 for a in results[T]['epoch_accs']]
        plt.plot(epochs, accs, marker='o', markersize=3, label=f'T={T}')
    plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy (%)')
    plt.title(f'[{dataset_name}] Validation Accuracy Curves for Different T Values')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "accuracy_curves_comparison.png", dpi=150)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="TemporalConv T-sweep (CUDA)")
    p.add_argument("--t-values", type=int, nargs='+', default=DEFAULT_T_VALUES,
                   help="要扫描的 T 值列表（默认 1 2 3 5 10）")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                   help="每个 T 值训练的 epoch 数（默认 50）")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="批大小（默认 64；多卡可设更大）")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker 数（默认 4）")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="CosineAnnealingLR 的最小学习率 eta_min（默认 1e-6）")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   choices=list(DATASET_REGISTRY.keys()),
                   help=f"要训练的任务数据集（默认三个都跑: {DEFAULT_DATASETS}）。"
                        f"可选: {list(DATASET_REGISTRY.keys())}。"
                        f"示例: --datasets LeftRight  或  --datasets LeftRight RightFeet")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    check_cuda_or_die()
    run_experiments(t_values=args.t_values,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    lr_min=args.lr_min,
                    datasets=args.datasets)
