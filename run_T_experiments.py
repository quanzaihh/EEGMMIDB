"""
TemporalConv T值对比实验脚本
测试不同时间窗口大小 T=1,2,3,5,10 对模型准确率的影响

实验配置：
- spike_ts: 160 (固定)
- epochs: 50
- batch_size: 64
- 数据集: EEGDataset2DLeftRight (左右手运动想象)
"""

import sys
sys.path.append("../utils/")
import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import random

from snn_T_experiment import WrapCUBASpikingCNN
from dataset import ToTensor, EEGDataset2DLeftRight, EEGAugmentor
from utility import train_validate_split_subjects, samples_per_class


# ============================================================
# 实验配置
# ============================================================
T_VALUES = [1, 2, 3, 5, 10]  # 时间窗口大小
EPOCHS = 50
SPIKE_TS = 160  # 固定
BATCH_SIZE = 64
SEED = 4

# 学习率
WT_LR = 0.0001
TS_LR = 0.0001
NEURON_LR = 0.0001

# 神经元参数
PARAM_LIST = [0.1, 0.1, 0.1, 0.3, 0.01, 0.1, 0.01]

# 数据集参数
import os
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

# 10折交叉验证分组
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

# 输出目录
RESULTS_DIR = Path("results_T_experiment")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 辅助函数
# ============================================================
def setup_logger(T):
    """为每个T值设置独立的日志"""
    log_dir = RESULTS_DIR / f"T{T}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f'T_{T}')
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
    """设置随机种子"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def test_accuracy(network, test_loader, device):
    """计算测试准确率"""
    network.eval()
    with torch.no_grad():
        class_correct = None
        class_total = None
        all_correct = 0
        all_total = 0
        for data in test_loader:
            eeg_data, label = data
            eeg_data = eeg_data.to(device)
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
# 训练函数 (支持 temporal_kernel 参数)
# ============================================================
def train_network_with_T(temporal_kernel, dataset_kwargs, spike_ts, param_list,
                         validate_subject_list, batch_size, epoch, lr, logger, log_dir, seed=None):
    """
    训练网络，支持指定 temporal_kernel
    
    Args:
        temporal_kernel: 时间卷积窗口大小 T
        dataset_kwargs: 数据集参数
        spike_ts: 时间步数 (固定160)
        param_list: 神经元参数
        validate_subject_list: 验证集受试者列表
        batch_size: 批大小
        epoch: 训练轮数
        lr: 学习率 [neuron_lr, ts_lr, wt_lr]
        logger: 日志记录器
        log_dir: 日志目录
        seed: 随机种子
    
    Returns:
        model, metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    
    if seed is not None:
        set_seed(int(seed))
        logger.info('Set random seed to: %d', int(seed))
    
    # 创建模型 (传入 temporal_kernel)
    net = WrapCUBASpikingCNN(spike_ts, device, param_list=param_list, temporal_kernel=temporal_kernel)
    net = net.to(device)
    if use_cuda:
        net = nn.DataParallel(net)
    net.train()
    
    # 统计参数量
    base_net = net.module if isinstance(net, nn.DataParallel) else net
    total_params, trainable_params = base_net.count_parameters()
    tc_params = base_net.count_temporal_conv_parameters()
    logger.info(f'Temporal Kernel T = {temporal_kernel}')
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'TemporalConv parameters: {tc_params:,}')
    
    # 数据集
    train_ds_kwargs = dataset_kwargs.copy()
    train_ds_kwargs.setdefault("transform", ToTensor())
    train_ds = EEGDataset2DLeftRight(**train_ds_kwargs)
    
    val_ds_kwargs = dataset_kwargs.copy()
    val_ds_kwargs.setdefault("transform", ToTensor())
    val_ds = EEGDataset2DLeftRight(**val_ds_kwargs)
    
    train_indices, val_indices = train_validate_split_subjects(train_ds, validate_subject_list)
    
    logger.info("Training Samples per Class:")
    samples_per_class(train_ds.label[train_indices])
    logger.info("Validate Samples per Class:")
    samples_per_class(val_ds.label[val_indices])
    
    train_sampler = sampler.SubsetRandomSampler(train_indices)
    val_sampler = sampler.SubsetRandomSampler(val_indices)
    
    loader_workers = 4 if use_cuda else 0
    loader_pin = use_cuda
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              sampler=train_sampler, num_workers=loader_workers, pin_memory=loader_pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=loader_workers, pin_memory=loader_pin)
    
    # 优化器
    try:
        decays = ['module.snn.c1_vdecay', 'module.snn.c2_vdecay', 'module.snn.c3_vdecay',
                  'module.snn.tc1_vdecay', 'module.snn.tc1_cdecay', 'module.snn.r1_vdecay',
                  'module.snn.f1_vdecay', 'module.snn.c1_cdecay', 'module.snn.c2_cdecay',
                  'module.snn.c3_cdecay', 'module.snn.r1_cdecay', 'module.snn.f1_cdecay']
        ts_weights = ['module.snn.ts_weights']
        decay_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in decays, net.named_parameters()))))
        ts_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in ts_weights, net.named_parameters()))))
        weights = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in decays + ts_weights, net.named_parameters()))))
        optimizer = optim.Adam([{'params': weights}, {'params': decay_params, 'lr': lr[0]}, {'params': ts_params, 'lr': lr[1]}], lr=lr[2])
    except Exception:
        optimizer = optim.Adam(net.parameters(), lr=lr[2] if len(lr) > 2 else 1e-4)
    
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                                       min_lr=1e-8)
    
    # 训练循环
    epoch_accs = []
    epoch_losses = []
    epoch_class_accs = []
    test_best_acc = 0.0
    best_epoch = 0
    
    for e in range(epoch):
        running_loss = 0.0
        iters = 0
        for i, data in enumerate(train_loader, 0):
            eeg_data, label = data
            eeg_data = eeg_data.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            output = net(eeg_data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            iters += 1
        
        net.eval()
        acc, class_acc = test_accuracy(net, val_loader, device)
        net.train()
        
        # 保存最佳模型
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
        
        logger.info('Epoch: %d, Loss: %.6f, Val Acc: %.3f %%', e, avg_loss, acc * 100)
        
        scheduler.step(acc)
    
    # 保存曲线图
    plt.figure(figsize=(10, 5))
    epochs_arr = np.arange(1, len(epoch_accs) + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr, epoch_accs, marker='o')
    plt.title(f'Validation Accuracy (T={temporal_kernel})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_arr, epoch_losses, marker='x', color='red')
    plt.title(f'Training Loss (T={temporal_kernel})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
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
# 主实验流程
# ============================================================
def run_experiments():
    """运行所有T值的实验"""
    results = {}
    
    print("\n" + "=" * 60)
    print("TemporalConv T值对比实验")
    print(f"T值: {T_VALUES}")
    print(f"Epochs: {EPOCHS}, Spike TS: {SPIKE_TS}")
    print("=" * 60 + "\n")
    
    for T in T_VALUES:
        print(f"\n{'='*60}")
        print(f"开始训练 T={T}")
        print("="*60)
        
        logger, log_dir = setup_logger(T)
        
        start_time = time.time()
        model, metrics = train_network_with_T(
            temporal_kernel=T,
            dataset_kwargs=ds_params,
            spike_ts=SPIKE_TS,
            param_list=PARAM_LIST,
            validate_subject_list=val_list[0],  # 使用第一折
            batch_size=BATCH_SIZE,
            epoch=EPOCHS,
            lr=[NEURON_LR, TS_LR, WT_LR],
            logger=logger,
            log_dir=log_dir,
            seed=SEED
        )
        train_time = time.time() - start_time
        
        results[T] = {
            'best_val_acc': metrics['best_acc'],
            'best_epoch': metrics['best_epoch'],
            'final_val_acc': metrics['epoch_accs'][-1],
            'train_time_sec': train_time,
            'train_time_min': train_time / 60,
            'total_params': metrics['total_params'],
            'tc_params': metrics['tc_params'],
            'epoch_accs': metrics['epoch_accs'],
            'epoch_losses': metrics['epoch_losses']
        }
        
        print(f"\nT={T} 完成:")
        print(f"  最佳准确率: {metrics['best_acc']*100:.2f}% (Epoch {metrics['best_epoch']})")
        print(f"  训练时间: {train_time/60:.1f} 分钟")
        print(f"  TemporalConv参数量: {metrics['tc_params']:,}")
    
    # 保存汇总结果
    results_file = RESULTS_DIR / "results_summary.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印汇总表格
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    print(f"{'T':<5} {'Best Acc':<12} {'Best Epoch':<12} {'Train Time':<12} {'TC Params':<12}")
    print("-" * 70)
    for T, r in results.items():
        print(f"{T:<5} {r['best_val_acc']*100:.2f}%{'':>5} {r['best_epoch']:<12} {r['train_time_min']:.1f} min{'':>4} {r['tc_params']:,}")
    
    # 生成对比图
    plot_comparison(results)
    
    print(f"\n结果已保存到: {RESULTS_DIR}")
    return results


def plot_comparison(results):
    """生成对比图"""
    T_values = list(results.keys())
    best_accs = [results[T]['best_val_acc'] * 100 for T in T_values]
    train_times = [results[T]['train_time_min'] for T in T_values]
    tc_params = [results[T]['tc_params'] for T in T_values]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 准确率对比
    axes[0].bar([str(t) for t in T_values], best_accs, color='steelblue')
    axes[0].set_xlabel('Temporal Kernel (T)')
    axes[0].set_ylabel('Best Validation Accuracy (%)')
    axes[0].set_title('Accuracy vs T')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(best_accs):
        axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    # 训练时间对比
    axes[1].bar([str(t) for t in T_values], train_times, color='coral')
    axes[1].set_xlabel('Temporal Kernel (T)')
    axes[1].set_ylabel('Training Time (min)')
    axes[1].set_title('Training Time vs T')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(train_times):
        axes[1].text(i, v + 0.5, f'{v:.1f}', ha='center')
    
    # 参数量对比
    axes[2].bar([str(t) for t in T_values], tc_params, color='green')
    axes[2].set_xlabel('Temporal Kernel (T)')
    axes[2].set_ylabel('TemporalConv Parameters')
    axes[2].set_title('Parameters vs T')
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(tc_params):
        axes[2].text(i, v, f'{v//1000}K', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_plot.png", dpi=150)
    plt.close()
    
    # 准确率曲线对比
    plt.figure(figsize=(10, 6))
    for T in T_values:
        epochs = range(1, len(results[T]['epoch_accs']) + 1)
        accs = [a * 100 for a in results[T]['epoch_accs']]
        plt.plot(epochs, accs, marker='o', markersize=3, label=f'T={T}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Curves for Different T Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(RESULTS_DIR / "accuracy_curves_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    results = run_experiments()
