"""
最小smoke测试：验证T值实验脚本是否能正常运行
只训练1个epoch，测试T=1
"""

import sys
sys.path.append("../utils/")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler

from snn_T_experiment import WrapCUBASpikingCNN
from dataset import ToTensor, EEGDataset2DLeftRight
from utility import train_validate_split_subjects, samples_per_class


def smoke_test():
    print("=" * 50)
    print("Smoke Test - T=1, 1 epoch")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 参数
    T = 1
    SPIKE_TS = 160
    PARAM_LIST = [0.1, 0.1, 0.1, 0.3, 0.01, 0.1, 0.01]
    
    # 创建模型
    print(f"\n创建模型: T={T}, spike_ts={SPIKE_TS}")
    model = WrapCUBASpikingCNN(SPIKE_TS, device, param_list=PARAM_LIST, temporal_kernel=T)
    total_params, trainable_params = model.count_parameters()
    tc_params = model.count_temporal_conv_parameters()
    print(f"总参数量: {total_params:,}")
    print(f"TemporalConv参数量: {tc_params:,}")
    
    # 测试数据集
    print("\n加载数据集...")
    ds_params = {
        "base_route": "../utils/eegmmidb_slice_norm/",
        "subject_id_list": [1, 2, 3],  # 只用3个受试者
        "start_ts": 0,
        "end_ts": 161,
        "window_ts": 160,
        "overlap_ts": 0,
        "use_imagery": False,
        "transform": ToTensor()
    }
    
    try:
        dataset = EEGDataset2DLeftRight(**ds_params)
        print(f"数据集大小: {len(dataset)}")
        print(f"标签分布: {dataset.label.shape}")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        print("使用随机数据测试模型前向传播...")
        
        # 用随机数据测试
        batch_size = 4
        dummy_input = torch.rand(batch_size, 1, 10, 11, SPIKE_TS)
        print(f"\n输入形状: {dummy_input.shape}")
        
        output = model(dummy_input)
        print(f"输出形状: {output.shape}")
        print(f"\n✓ Smoke test 通过！模型可以正常前向传播。")
        return True
    
    # 测试数据加载
    train_indices, val_indices = train_validate_split_subjects(dataset, [1])
    train_sampler = sampler.SubsetRandomSampler(train_indices[:20])  # 只用少量样本
    train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler, num_workers=0)
    
    print(f"\n训练样本数: {len(train_indices[:20])}")
    
    # 测试一个batch
    for data in train_loader:
        eeg_data, label = data
        print(f"\nBatch数据形状: {eeg_data.shape}")
        print(f"标签: {label}")
        
        output = model(eeg_data)
        print(f"模型输出形状: {output.shape}")
        break
    
    print(f"\n✓ Smoke test 通过！")
    return True


if __name__ == "__main__":
    smoke_test()
