# T-sweep 实验 · CUDA 服务器使用说明

CUDA 专用运行脚本 `run_T_experiments_cuda.py` 用于在 GPU 服务器上跑 SNN CUBA-LIF
模型的 Temporal Kernel T 值对比实验（T ∈ {1, 2, 3, 5, 10}）。

## 1. 文件清单（需要拷贝到服务器同一目录下）

**必须**
```
run_T_experiments_cuda.py    # 入口脚本（CUDA 专用）
snn_T_experiment.py          # 模型定义（device-agnostic，用 self.device）
dataset.py                   # EEG 数据集加载
utility.py                   # split / samples_per_class 等工具
cbam.py cbam_ca.py cbam_sa.py  # CBAM 注意力模块
mean_channel.npy std_channel.npy  # 数据标准化用的均值/方差
eegmmidb_slice_norm/         # 数据集目录（S001 … S109）
```

**可选**（CPU 本地跑用，服务器上不需要）
```
run_T_experiments.py
```

## 2. 环境依赖

```bash
# Python 3.10+ 推荐
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib mne scipy
```

PyTorch 版本需要和服务器 CUDA 匹配：
- CUDA 11.8 → `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.1 → `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- CUDA 12.4 → `pip install torch --index-url https://download.pytorch.org/whl/cu124`

## 3. 启动方式

### 3.1 全量扫描（默认配置）
```bash
python run_T_experiments_cuda.py
```
等价于：T=[1,2,3,5,10]，epochs=50，batch=64，num_workers=4。

### 3.2 指定 GPU
```bash
CUDA_VISIBLE_DEVICES=2 python run_T_experiments_cuda.py
```

### 3.3 多卡 DataParallel
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run_T_experiments_cuda.py --batch-size 256
```
脚本会自动用 `nn.DataParallel` 跨可见 GPU 切分。

### 3.4 快速消融（缩水跑法）
```bash
# 只跑 T=1, 3, 10；每个 10 epoch
python run_T_experiments_cuda.py --t-values 1 3 10 --epochs 10 --batch-size 128
```

### 3.5 后台 + 日志
```bash
nohup python -u run_T_experiments_cuda.py \
    > results_T_experiment_cuda/run.log 2>&1 &
echo "PID=$!"
tail -f results_T_experiment_cuda/run.log
```

## 4. 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--t-values` | `1 2 3 5 10` | 要扫描的 T 值列表 |
| `--epochs` | `50` | 每个 T 值的训练 epoch 数 |
| `--batch-size` | `64` | 批大小（多卡可放大） |
| `--num-workers` | `4` | DataLoader worker 数 |

## 5. 输出物

所有结果落在 `results_T_experiment_cuda/`：

```
results_T_experiment_cuda/
├── T1/                          # 每个 T 值一个子目录
│   ├── train.log                # 纯文本训练日志
│   ├── training_curve_T1.png    # 单 T 的 acc/loss 曲线
│   └── checkpoints/
│       └── checkpoint_best.pth  # 当前最佳验证精度的权重
├── T2/ T3/ T5/ T10/             # 同上
├── results_summary.json         # 所有 T 值汇总指标
├── comparison_plot.png          # acc / time / params 三联柱状图
└── accuracy_curves_comparison.png  # 5 条验证精度曲线叠图
```

`results_summary.json` 里对每个 T 记录：
- `best_val_acc` / `best_epoch`
- `train_time_sec` / `train_time_min`
- `total_params` / `tc_params`
- `epoch_accs` / `epoch_losses`（逐 epoch 序列）

## 6. 与 CPU 版本 (`run_T_experiments.py`) 的差异

| 项 | CPU 版 | CUDA 版 |
|---|---|---|
| 设备选择 | 自动降级到 CPU | 启动时强校验 CUDA，否则 `sys.exit(1)` |
| DataParallel | 禁用 | 自动按 `torch.cuda.device_count()` 启用 |
| `pin_memory` | `False` | `True` |
| `num_workers` | `0`（macOS fork 问题） | `4`（可 `--num-workers` 调整） |
| `persistent_workers` | 不用 | `True`（避免 epoch 间重建 worker） |
| `cudnn.benchmark` | N/A | 因确定性需求设为 `False`，若求速可改 `True` |
| 命令行参数 | 无 | 支持 `--t-values / --epochs / --batch-size / --num-workers` |
| 日志 | 控制台 | 控制台 + 每个 T 独立文件 `train.log` |
| GPU 内存监控 | 无 | 每 epoch 打印 `max_memory_allocated` 并重置 |
| 结果目录 | `results_T_experiment/` | `results_T_experiment_cuda/` |

## 7. 性能参考（粗估）

在单张 RTX 3090 / A100 上，冒烟测试外推：

| T 值 | TC 参数 | 估算单 epoch 耗时（全量 103 subjects） |
|---|---|---|
| 1  | 66K   | ~10 s |
| 2  | 132K  | ~15 s |
| 3  | 198K  | ~20 s |
| 5  | 330K  | ~30 s |
| 10 | 658K  | ~60 s |

**全量 50 epochs × 5 T 值 ≈ 2~3 小时**（单卡 A100，仅供参考）。多卡 batch 放大后可再缩短。

## 8. 常见问题排查

### `CUDA out of memory`
```bash
python run_T_experiments_cuda.py --batch-size 32
```

### `num_workers > 0` 导致 DataLoader 卡住
```bash
python run_T_experiments_cuda.py --num-workers 0
```

### 想复现确定性结果
代码里已经固定 `SEED=4` 并启用 `cudnn.deterministic=True, cudnn.benchmark=False`。
相同硬件 + 相同 PyTorch 版本应可逐 step 复现。

### 只做某几折的验证
改 `run_T_experiments_cuda.py` 里 `validate_subject_list=val_list[0]`，把下标换成
`val_list[1]..val_list[9]` 跑其它折，或改为循环跑 10 折做交叉验证。

## 9. 中断恢复

脚本尚不支持自动 resume。如果中途挂掉：
- 已完成的 T 值在 `results_T_experiment_cuda/T{K}/checkpoints/checkpoint_best.pth` 里
- 重跑时用 `--t-values` 只跑剩下的 T 值即可，例如：
  ```bash
  # 假设 T=1,2 已跑完
  python run_T_experiments_cuda.py --t-values 3 5 10
  ```
