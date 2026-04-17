"""
SNN模型 - 参数化temporal_kernel版本
用于实验对比不同时间窗口大小 T=1,2,3,5,10 的效果
"""

import os
import sys

sys.path.append("../utils/")
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
from torch.distributions.bernoulli import Bernoulli
import torch.optim as optim
import matplotlib.pyplot as plt
import cbam


DROPOUT_TC = 0.3
DROPOUT_REC = 0.3
DROPOUT_FC = 0.3
AMP = 0.3


class PseudoSpikeRect(torch.autograd.Function):
    """Define custom autograd function for Spike Function """
    
    @staticmethod
    def forward(ctx, input, vth, grad_win):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.grad_win = grad_win
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        grad_win = ctx.grad_win
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - vth) < grad_win
        return AMP * grad_input * spike_pseudo_grad.float(), None, None


class PseudoSpikeRectDropout(torch.autograd.Function):
    """Define custom autograd function for Spike Function with dropout"""
    
    @staticmethod
    def forward(ctx, input, vth, grad_win, mask):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.grad_win = grad_win
        ctx.mask = mask
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        grad_win = ctx.grad_win
        mask = ctx.mask
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - vth) < grad_win
        spike_pseudo_grad[mask==0] = 0
        return AMP * grad_input * spike_pseudo_grad.float(), None, None, None


class FeedForwardCUBALIFCell(nn.Module):
    def __init__(self, psp_func, pseudo_grad_ops, param, record_neuron=None):
        """
        :param psp_func: pre-synaptic function
        :param pseudo_grad_ops: pseudo gradient operation
        :param param: (current decay, voltage decay, voltage threshold, gradient window parameter)
        """
        super(FeedForwardCUBALIFCell, self).__init__()
        self.psp_func = psp_func
        self.pseudo_grad_ops = pseudo_grad_ops
        self.cdecay, self.vdecay, self.vth, self.grad_win = param
        self.record_neuron = record_neuron
        self.neuron_current = []
        self.neuron_voltages = []
        self.neuron_spikes = []
        self.summ_pre = None

    def forward(self, input_data, state):
        """
        :param input_data: input spike from pre-synaptic neurons
        :param state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        :return: output spike, (output spike, current, voltage)
        """
        pre_spike, pre_current, pre_volt = state
        summ_pre = self.psp_func(input_data)
        current = self.cdecay * pre_current + summ_pre
        volt = self.vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win)
        return output, (output, current, volt)


class FeedForwardCUBALIFCellDropout(nn.Module):
    def __init__(self, psp_func, pseudo_grad_ops, param):
        """
        :param psp_func: pre-synaptic function
        :param pseudo_grad_ops: pseudo gradient operation
        :param param: (current decay, voltage decay, voltage threshold, gradient window parameter)
        """
        super(FeedForwardCUBALIFCellDropout, self).__init__()
        self.psp_func = psp_func
        self.pseudo_grad_ops = pseudo_grad_ops
        self.cdecay, self.vdecay, self.vth, self.grad_win = param

    def forward(self, input_data, state, mask, train):
        """
        :param input_data: input spike from pre-synaptic neurons
        :param state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        :param mask: dropout mask
        :param train: training mode
        :return: output spike, (output spike, current, voltage)
        """
        pre_spike, pre_current, pre_volt = state
        current = self.cdecay * pre_current + self.psp_func(input_data)
        if train is True:
            current = current * mask
        volt = self.vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win, mask)
        return output, (output, current, volt)


class TemporalConvCUBALIFCell(nn.Module):
    def __init__(self, kernel, psp_func_list, pseudo_grad_ops, param):
        """
        :param kernel: kernel size of temporal conv (时间窗口大小 T)
        :param psp_func_list: list of psp functions (same number as kernel)
        :param pseudo_grad_ops: pseudo gradient operation
        :param param: (current decay, voltage decay, voltage threshold, gradient window parameter)
        """
        super(TemporalConvCUBALIFCell, self).__init__()
        self.kernel = kernel
        self.psp_func_list = nn.ModuleList(psp_func_list)
        self.pseudo_grad_ops = pseudo_grad_ops
        self.cdecay, self.vdecay, self.vth, self.grad_win = param
        self.record_inputs = False
        self.recorded_inputs = []

    def forward(self, input_data_list, state, mask, train, if_output=False):
        """
        :param input_data_list: list of input spike from different timesteps (same number as kernel)
        :param state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        :param mask: dropout mask
        :param train: training mode
        :return: output spike, (output spike, current, voltage)
        """
        pre_spike, pre_current, pre_volt = state
        current = self.cdecay * pre_current
        pre_current = []
        linear_output = None
        if len(input_data_list) == self.kernel:
            for ts in range(self.kernel):
                linear_output = self.psp_func_list[ts](input_data_list[ts])
                current += linear_output
                pre_current.append(linear_output)

        if len(input_data_list) < self.kernel:
            ts_diff = self.kernel - len(input_data_list)
            for ts in range(len(input_data_list)):
                linear_output = self.psp_func_list[ts + ts_diff](input_data_list[ts])
                current += linear_output
                pre_current.append(linear_output)

        if train is True:
            current = current * mask  
        volt = self.vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win, mask)
        
        if getattr(self, 'record_inputs', False) and not train:
            try:
                cpu_list = [x.detach().cpu().numpy() if hasattr(x, 'detach') else np.array(x) for x in input_data_list]
                self.recorded_inputs.append(cpu_list)
            except Exception:
                pass
        if if_output:
            return output, (output, current, volt), pre_current
        else:
            return output, (output, current, volt)


class CUBASpikingCNN(nn.Module):
    def __init__(self, spike_ts, params, temporal_kernel=3, record_neuron=None, device=None):
        """
        :param spike_ts: spike timesteps (固定为160)
        :param params: list of param for each neuron layer (cdecay, vdecay, vth, grad_win, th_amp, th_decay, base_th)
        :param temporal_kernel: 时间卷积的窗口大小 T (默认3，可配置为1,2,3,5,10)
        :param record_neuron: 目标神经元坐标
        :param device: 设备 (cuda/cpu)
        """
        super(CUBASpikingCNN, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.spike_ts = spike_ts
        self.temporal_kernel = temporal_kernel  # 参数化时间窗口大小
        
        self.cdecay, self.vdecay, self.vth, self.grad_win, self.th_amp, self.th_decay, self.base_th = params
        pseudo_grad_ops = PseudoSpikeRect.apply
        pseudo_grad_ops_rec = PseudoSpikeRectDropout.apply

        self.c1_vdecay = nn.Parameter(torch.ones(1, 64, 8, 9, device=self.device) * self.vdecay)
        self.c2_vdecay = nn.Parameter(torch.ones(1, 128, 6, 7, device=self.device) * self.vdecay)
        self.c3_vdecay = nn.Parameter(torch.ones(1, 256, 1, 1, device=self.device) * self.vdecay)
        self.tc1_vdecay = nn.Parameter(torch.ones(1, 256, device=self.device) * self.vdecay)
        self.f1_vdecay = nn.Parameter(torch.ones(1, 256, device=self.device) * self.vdecay)

        self.c1_cdecay = nn.Parameter(torch.ones(1, 64, 8, 9, device=self.device) * self.cdecay)
        self.c2_cdecay = nn.Parameter(torch.ones(1, 128, 6, 7, device=self.device) * self.cdecay)
        self.c3_cdecay = nn.Parameter(torch.ones(1, 256, 1, 1, device=self.device) * self.cdecay)
        self.tc1_cdecay = nn.Parameter(torch.ones(1, 256, device=self.device) * self.vdecay)
        self.f1_cdecay = nn.Parameter(torch.ones(1, 256, device=self.device) * self.cdecay)

        self.ts_weights = nn.Parameter(torch.ones((self.spike_ts), device=self.device) / self.spike_ts)
        
        self.conv1 = FeedForwardCUBALIFCell(
            nn.Conv2d(1, 64, (3, 3), bias=True), 
            pseudo_grad_ops, 
            [self.c1_cdecay, self.c1_vdecay, self.vth, self.grad_win],
            record_neuron=record_neuron
        )

        self.conv2 = FeedForwardCUBALIFCell(
            nn.Conv2d(64, 128, (3, 3), bias=True), 
            pseudo_grad_ops,
            [self.c2_cdecay, self.c2_vdecay, self.vth, self.grad_win], 
            record_neuron=None
        )
        
        self.avg_pool = nn.AvgPool2d(2)

        self.conv3 = FeedForwardCUBALIFCell(
            nn.Conv2d(128, 256, (3, 3), bias=True), 
            pseudo_grad_ops, 
            [self.c3_cdecay, self.c3_vdecay, self.vth, self.grad_win], 
            record_neuron=None
        )

        # CBAM注意力模块
        self.cbam_conv1 = cbam.CBAM(gate_channels=64)
        self.cbam_conv2 = cbam.CBAM(gate_channels=128)
        self.cbam_conv3 = cbam.CBAM(gate_channels=256)

        # 关键修改：参数化temporal_kernel
        self.temp_conv1 = TemporalConvCUBALIFCell(
            kernel=temporal_kernel,  # 参数化
            psp_func_list=[nn.Linear(256, 256, bias=True) for _ in range(temporal_kernel)],  # 数量=kernel
            pseudo_grad_ops=pseudo_grad_ops_rec,
            param=[self.tc1_cdecay, self.tc1_vdecay, self.vth, self.grad_win]
        )

        self.fc1 = FeedForwardCUBALIFCellDropout(
            nn.Linear(256, 256, bias=True), 
            pseudo_grad_ops_rec, 
            [self.f1_cdecay, self.f1_vdecay, self.vth, self.grad_win]
        )
        
        self.fc2 = nn.Linear(256, 2, bias=False)

    def forward(self, input_data, states, if_output=False):
        """
        :param input_data: input EEG spike trains
        :param states: list of (init spike, init voltage)
        :return: output
        """
        batch_size = input_data.shape[0]
        output_spikes = []
        temp_conv_spike_buffer = []
        dropout_tc = DROPOUT_TC
        dropout_rec = DROPOUT_REC
        dropout_fc = DROPOUT_FC
        c1_state, c2_state, c3_state, tc1_state, f1_state = \
            states[0], states[1], states[2], states[3], states[4]

        total_current = []
        
        mask_tc = Bernoulli(
            torch.full_like(torch.zeros(batch_size, 256, device=self.device), 1 - dropout_tc)).sample() / (
                       1 - dropout_tc)

        mask = Bernoulli(
            torch.full_like(torch.zeros(batch_size, 256, device=self.device), 1 - dropout_rec)).sample() / (
                       1 - dropout_rec)

        mask_fc = Bernoulli(
            torch.full_like(torch.zeros(batch_size, 256, device=self.device), 1 - dropout_fc)).sample() / (
                       1 - dropout_fc)

        for step in range(self.spike_ts):
            state = []
            input_spike = input_data[:, :, :, :, step]
            c1_spike, c1_state = self.conv1(input_spike, c1_state)
            state.append([c1_state[0][:, 0, :, :], c1_state[1][:, 0, :, :], c1_state[2][:, 0, :, :]])
            
            # CBAM after conv1
            c1_spike_reshaped = c1_spike.view(batch_size, 64, -1)
            c1_spike = self.cbam_conv1(c1_spike_reshaped).view(batch_size, 64, 8, 9)
            
            c2_spike, c2_state = self.conv2(c1_spike, c2_state)
            # CBAM after conv2
            c2_spike_reshaped = c2_spike.view(batch_size, 128, -1)
            c2_spike = self.cbam_conv2(c2_spike_reshaped).view(batch_size, 128, 6, 7)

            avg_pool_c2_spike = self.avg_pool(c2_spike)

            c3_spike, c3_state = self.conv3(avg_pool_c2_spike, c3_state)
            # CBAM after conv3
            c3_spike_reshaped = c3_spike.view(batch_size, 256, -1)
            c3_spike_att = self.cbam_conv3(c3_spike_reshaped)
            c3_spike = c3_spike_att.view(batch_size, 256, 1, 1)
            
            flat_c3_spike = c3_spike.view(batch_size, -1)

            temp_conv_spike_buffer.append(flat_c3_spike)
            # 关键修改：使用参数化的temporal_kernel
            if len(temp_conv_spike_buffer) > self.temporal_kernel:
                temp_conv_spike_buffer.pop(0)

            # temporal conv over recent buffers
            pre_currents = []
            if if_output:
                tc1_spike, tc1_state, pre_currents = self.temp_conv1(temp_conv_spike_buffer, tc1_state, mask_tc, self.training, if_output)
            else:
                tc1_spike, tc1_state = self.temp_conv1(temp_conv_spike_buffer, tc1_state, mask_tc, self.training)

            state.append([tc1_state, pre_currents])
            total_current.append(state)

            # final FCs
            f1_spike, f1_state = self.fc1(tc1_spike, f1_state, mask_fc, self.training)
            f2_output = self.fc2(f1_spike)
            output_spikes += [f2_output * self.ts_weights[step]]
        outputs = torch.stack(output_spikes).sum(dim=0)

        if if_output:
            return outputs, total_current
        else:
            return outputs


class WrapCUBASpikingCNN(nn.Module):
    def __init__(self, spike_ts, device, param_list, temporal_kernel=3, record_neuron=None):
        """
        :param spike_ts: spike timesteps (固定160)
        :param device: device
        :param param_list: list of param for each neuron layer
        :param temporal_kernel: 时间卷积窗口大小 T (默认3)
        :param record_neuron: 目标神经元坐标
        """
        super(WrapCUBASpikingCNN, self).__init__()
        self.device = device
        self.temporal_kernel = temporal_kernel
        self.cdecay, self.vdecay, self.vth, self.grad_win, self.th_amp, self.th_decay, self.base_th = param_list
        self.snn = CUBASpikingCNN(spike_ts, param_list, temporal_kernel=temporal_kernel, record_neuron=record_neuron, device=device)

    def forward(self, input_data, if_output=False):
        """
        :param input_data: input EEG spike trains
        :return: output
        """
        batch_size = input_data.shape[0]
        c1_current = torch.zeros(batch_size, 64, 8, 9, device=self.device)
        c1_volt = torch.zeros(batch_size, 64, 8, 9, device=self.device)
        c1_spike = torch.zeros(batch_size, 64, 8, 9, device=self.device)
        c1_state = (c1_spike, c1_current, c1_volt)
        c2_current = torch.zeros(batch_size, 128, 6, 7, device=self.device)
        c2_volt = torch.zeros(batch_size, 128, 6, 7, device=self.device)
        c2_spike = torch.zeros(batch_size, 128, 6, 7, device=self.device)
        c2_state = (c2_spike, c2_current, c2_volt)
        c3_current = torch.zeros(batch_size, 256, 1, 1, device=self.device)
        c3_volt = torch.zeros(batch_size, 256, 1, 1, device=self.device)
        c3_spike = torch.zeros(batch_size, 256, 1, 1, device=self.device)
        c3_state = (c3_spike, c3_current, c3_volt)
        tc1_current = torch.zeros(batch_size, 256, device=self.device)
        tc1_volt = torch.zeros(batch_size, 256, device=self.device)
        tc1_spike = torch.zeros(batch_size, 256, device=self.device)
        tc1_state = (tc1_spike, tc1_current, tc1_volt)
        f1_current = torch.zeros(batch_size, 256, device=self.device)
        f1_volt = torch.zeros(batch_size, 256, device=self.device)
        f1_spike = torch.zeros(batch_size, 256, device=self.device)
        f1_state = (f1_spike, f1_current, f1_volt)
        states = (c1_state, c2_state, c3_state, tc1_state, f1_state)
        if if_output:
            [output, spike] = self.snn(input_data, states, if_output)
            return [output, spike]
        else:
            output = self.snn(input_data, states, if_output)
            return output
    
    def count_parameters(self):
        """统计模型参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def count_temporal_conv_parameters(self):
        """统计TemporalConv模块参数量"""
        tc_params = sum(p.numel() for p in self.snn.temp_conv1.parameters())
        return tc_params
