import torch
from torch.utils.data import Dataset
from utility import get_eeg_2d_map, generate_left_right_dataset_from_saved_data,\
    generate_left_feet_classes_dataset_from_saved_data, generate_right_feet_classes_dataset_from_saved_data
import numpy as np


class EEGDataset2DLeftRight(Dataset):
    """ Reguar Dataset for EEGMMIDB data """
    def __init__(self, base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                 use_imagery, transform=None):
        """
        :param base_route: base route to EEGMMIDB data
        :param subject_id_list: list of subjects
        :param start_ts: start timestep for each slice for epoch
        :param end_ts: maximum end timestep for each slice for epoch
        :param window_ts: window timestep for each epoch
        :param overlap_ts: overlap timestep between two window
        :param use_imagery: if true use imagery data instead of movement data
        :param transform: optional transform to be applied on a sample
        """
        self.data, self.label, self.epoch_ts, self.epoch_subjects = generate_left_right_dataset_from_saved_data(
            base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
            use_imagery, get_eeg_2d_map(), (10, 11)
        )

        self.transform = transform

    def __len__(self):
        """
        :return: length of the entire dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        Get one entry of data by item
        :param item: index of data
        :return: data with label
        """
        item_label = self.label[item]
        item_data = self.data[item, :, :].reshape(10, 11, -1, 1)
        if self.transform:
            item_data = self.transform(item_data)
        item_data_w_label = [item_data, item_label]
        return item_data_w_label


class EEGDatasetLeftFeet(Dataset):
    """ Reguar Dataset for EEGMMIDB data """
    def __init__(self, base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                 use_imagery, transform=None):
        """
        :param base_route: base route to EEGMMIDB data
        :param subject_id_list: list of subjects
        :param start_ts: start timestep for each slice for epoch
        :param end_ts: maximum end timestep for each slice for epoch
        :param window_ts: window timestep for each epoch
        :param overlap_ts: overlap timestep between two window
        :param use_imagery: if true use imagery data instead of movement data
        :param use_no_movement: if true use no movement as class 0 in movement data
        :param transform: optional transform to be applied on a sample
        """
        self.data, self.label, self.epoch_ts, self.epoch_subjects = generate_left_feet_classes_dataset_from_saved_data(
            base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
            use_imagery, get_eeg_2d_map(), (10, 11)
        )
        self.transform = transform

    def __len__(self):
        """
        :return: length of the entire dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        Get one entry of data by item
        :param item: index of data
        :return: data with label
        """
        item_label = self.label[item]
        item_data = self.data[item, :, :].reshape(10, 11, -1, 1)
        if self.transform:
            item_data = self.transform(item_data)
        item_data_w_label = [item_data, item_label]
        return item_data_w_label


class EEGDatasetRightFeet(Dataset):
    """ Reguar Dataset for EEGMMIDB data """
    def __init__(self, base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
                 use_imagery, transform=None):
        """
        :param base_route: base route to EEGMMIDB data
        :param subject_id_list: list of subjects
        :param start_ts: start timestep for each slice for epoch
        :param end_ts: maximum end timestep for each slice for epoch
        :param window_ts: window timestep for each epoch
        :param overlap_ts: overlap timestep between two window
        :param use_imagery: if true use imagery data instead of movement data
        :param transform: optional transform to be applied on a sample
        """
        self.data, self.label, self.epoch_ts, self.epoch_subjects = generate_right_feet_classes_dataset_from_saved_data(
            base_route, subject_id_list, start_ts, end_ts, window_ts, overlap_ts,
            use_imagery, get_eeg_2d_map(), (10, 11)
        )
        self.transform = transform

    def __len__(self):
        """
        :return: length of the entire dataset
        """
        return self.data.shape[0]

    def __getitem__(self, item):
        """
        Get one entry of data by item
        :param item: index of data
        :return: data with label
        """
        item_label = self.label[item]
        item_data = self.data[item, :, :].reshape(10, 11, -1, 1)
        if self.transform:
            item_data = self.transform(item_data)
        item_data_w_label = [item_data, item_label]
        return item_data_w_label


class ToTensor(object):
    """ Transformation to convert ndarray to pytorch tensor"""
    def __call__(self, sample):
        """
        :param sample: ndarray
        :return: pytorch tensor
        """
        sample = sample.transpose((3, 0, 1, 2))
        sample = torch.from_numpy(sample).float()
        return sample

class EEGAugmentor:
    """EEG数据增强工具类（适配时序/Spike数据）"""
    def __init__(self,
                 time_shift_range=2,    # 时间移位范围（时间步）
                 noise_std=0.01,        # 高斯噪声标准差
                 prob=0.5):             # 增强概率（避免过度增强）
        self.time_shift_range = time_shift_range
        self.noise_std = noise_std
        self.prob = prob

    def time_shift(self, data):
        """时间移位增强（最后一维为时间步）"""
        # data shape: [channels, height, width, time] 或 [batch, channels, height, width, time]
        shift = np.random.randint(-self.time_shift_range, self.time_shift_range + 1)
        if shift != 0:
            # 对最后一维（时间步）进行循环移位
            data = np.roll(data, shift, axis=2)
            # 可选：对移位后的边缘填充0（避免循环信息泄露）
            if shift > 0:
                data[..., :shift] = 0
            else:
                data[..., shift:] = 0
        return data

    def add_noise(self, data):
        """高斯噪声注入"""
        noise = np.random.normal(0, self.noise_std, size=data.shape)
        return data + noise

    def spike_dropout(self, data, dropout_rate=0.1):
        """脉冲随机丢弃（模拟神经元发放噪声）"""
        mask = np.random.binomial(1, 1 - dropout_rate, size=data.shape)
        return data * mask

    def __call__(self, data):
        if np.random.rand() < self.prob:
            data = self.time_shift(data)
        if np.random.rand() < self.prob:
            data = self.add_noise(data)
        if np.random.rand() < self.prob:  # 仅对Spike数据生效
            data = self.spike_dropout(data)
        return data