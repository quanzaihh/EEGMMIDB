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

# Define custom autograd function for Spike Function
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
        self.record_neuron = record_neuron  # 存储目标神经元坐标
        self.neuron_current = []  # 存储电流序列
        self.neuron_voltages = []  # 存储膜电位序列
        self.neuron_spikes = []    # 存储脉冲序列
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

        # 记录目标神经元的膜电位和脉冲（仅训练/测试时记录）
        # if self.record_neuron is not None and self.training is False:  # 测试模式下记录
        #     c, h, w = self.record_neuron
        #     # 提取当前时间步的膜电位和脉冲（取batch第0个样本）
        #
        #     self.neuron_current.append( summ_pre[0, c, h, w].item())  # 0表示第1个样本
        #     self.neuron_voltages.append(volt[0, c, h, w].item())  # 0表示第1个样本
        #     self.neuron_spikes.append(output[0, c, h, w].item())
        #     if len(self.neuron_spikes) == 160:  # 只保留最近100个时间步的数据
        #         plt.subplot(3, 1, 1)
        #         plt.plot(self.neuron_current)
        #         plt.subplot(3, 1, 2)
        #         plt.plot(self.neuron_voltages)
        #         plt.axhline(y=0.1, color='r', linestyle='--', linewidth=2, label='强度=0.1')  # 添加0.1的水平线
        #         plt.subplot(3, 1, 3)
        #         plt.plot(self.neuron_spikes)
        #         plt.savefig("neuron_dynamics.png", dpi=300, bbox_inches="tight")  # 保存图片（高清）
        #
        #         plt.tight_layout()
        #         plt.show()

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

class RecurrentCUBALIFCell(nn.Module):
    def __init__(self, psp_func, rec_func, pseudo_grad_ops, param):
        """
        :param psp_func: pre-synaptic function
        :param rec_func: recurrent connection function
        :param pseudo_grad_ops: pseudo gradient operation
        :param param: (current decay, voltage decay, voltage threshold, gradient window parameter)
        """
        super(RecurrentCUBALIFCell, self).__init__()
        self.psp_func = psp_func
        self.rec_func = rec_func
        self.pseudo_grad_ops = pseudo_grad_ops
        self.cdecay, self.vdecay, self.grad_win, self.th_amp, self.th_decay, self.base_th = param

    def forward(self, input_data, state, mask, train):
        """
        :param input_data: input spike from pre-synaptic neurons
        :param state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        :param mask: dropout mask
        :param train: training mode
        :return: output spike, (output spike, current, voltage)
        """
        pre_spike, pre_current, pre_volt, pre_vth = state
        current = self.cdecay * pre_current + self.psp_func(input_data) + self.rec_func(pre_spike)

        if train is True:
            current = current * mask 

        volt = self.vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, pre_vth, self.grad_win, mask)
        vth = ((pre_vth + self.th_amp) * output) + torch.clamp((pre_vth * self.th_decay), min=self.base_th) * (1. - output)
        return output, (output, current, volt, vth)


class TemporalConvCUBALIFCell(nn.Module):
    def __init__(self, kernel, psp_func_list, pseudo_grad_ops, param):
        """
        :param kernel: kernel size of temporal conv
        :param psp_func_list: list of psp functions (same number as kernel)
        :param pseudo_grad_ops: pseudo gradient operation
        :param param: (current decay, voltage decay, voltage threshold, gradient window parameter)
        """
        super(TemporalConvCUBALIFCell, self).__init__()
        self.kernel = kernel
        self.psp_func_list = nn.ModuleList(psp_func_list)
        self.pseudo_grad_ops = pseudo_grad_ops
        self.cdecay, self.vdecay, self.vth, self.grad_win = param

    def forward(self, input_data_list, state, mask, train):
        """
        :param input_data_list: list of input spike from different timesteps (same number as kernel)
        :param state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        :param mask: dropout mask
        :param train: training mode
        :return: output spike, (output spike, current, voltage)
        """
        pre_spike, pre_current, pre_volt = state
        current = self.cdecay * pre_current
        if len(input_data_list) == self.kernel:
            for ts in range(self.kernel):
                current += self.psp_func_list[ts](input_data_list[ts])

        if len(input_data_list) < self.kernel:
            ts_diff = self.kernel - len(input_data_list)
            for ts in range(len(input_data_list)):
                current += self.psp_func_list[ts + ts_diff](input_data_list[ts])

        if train is True:
            current = current * mask  
        volt = self.vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win, mask)
        return output, (output, current, volt)


class CUBASpikingCNN(nn.Module):
    def __init__(self, spike_ts, params, record_neuron=None):
        """
        :param spike_ts: spike timesteps
        :param param_list: list of param for each neuron layer (cdecay, vdecay, vth, grad_win, th_amp, th_decay, base_th)

        """
        super(CUBASpikingCNN, self).__init__()
        self.device = torch.device("cuda")
        self.spike_ts = spike_ts
        self.cdecay, self.vdecay, self.vth, self.grad_win, self.th_amp, self.th_decay, self.base_th = params
        pseudo_grad_ops = PseudoSpikeRect.apply
        pseudo_grad_ops_rec = PseudoSpikeRectDropout.apply

        self.c1_vdecay = nn.Parameter(torch.ones(1, 64, 8, 9, device=self.device) * self.vdecay)
        self.c2_vdecay = nn.Parameter(torch.ones(1, 128, 6, 7, device=self.device) * self.vdecay)
        self.c3_vdecay = nn.Parameter(torch.ones(1, 256, 1, 1, device=self.device) * self.vdecay)
        self.tc1_vdecay = nn.Parameter(torch.ones(1, 256, device=self.device) * self.vdecay)
        self.r1_vdecay = nn.Parameter(torch.ones(1, 256, device=self.device) * self.vdecay)
        self.f1_vdecay = nn.Parameter(torch.ones(1, 256, device=self.device) * self.vdecay)

        self.c1_cdecay = nn.Parameter(torch.ones(1, 64, 8, 9, device=self.device) * self.cdecay)
        self.c2_cdecay = nn.Parameter(torch.ones(1, 128, 6, 7, device=self.device) * self.cdecay)
        self.c3_cdecay = nn.Parameter(torch.ones(1, 256, 1, 1, device=self.device) * self.cdecay)
        self.tc1_cdecay = nn.Parameter(torch.ones(1, 256, device=self.device) * self.vdecay)
        self.r1_cdecay = nn.Parameter(torch.ones(1, 256, device=self.device) * self.cdecay)
        self.f1_cdecay = nn.Parameter(torch.ones(1, 256, device=self.device) * self.cdecay)

        self.ts_weights = nn.Parameter(torch.ones((self.spike_ts), device=self.device) / self.spike_ts)
        
        self.conv1 = FeedForwardCUBALIFCell(nn.Conv2d(1, 64, (3, 3), bias=True), pseudo_grad_ops, 
        [self.c1_cdecay, self.c1_vdecay, self.vth, self.grad_win],
                            record_neuron  =   record_neuron # 传入目标神经元坐标
                                            )



        self.conv2 = FeedForwardCUBALIFCell(nn.Conv2d(64, 128, (3, 3), bias=True), pseudo_grad_ops, 
        [self.c2_cdecay, self.c2_vdecay, self.vth, self.grad_win], record_neuron=None)
        
        self.avg_pool = nn.AvgPool2d(2)
        
        self.conv3 = FeedForwardCUBALIFCell(nn.Conv2d(128, 256, (3, 3), bias=True), pseudo_grad_ops, 
        [self.c3_cdecay, self.c3_vdecay, self.vth, self.grad_win], record_neuron=None)

        # 初始化CBAM模块（根据插入位置选择）
        self.cbam_conv1 = cbam.CBAM(gate_channels=64)  # conv1输出通道数64
        self.cbam_conv3 = cbam.CBAM(gate_channels=256) # conv3输出通道数256


        self.temp_conv1 = TemporalConvCUBALIFCell(3, [nn.Linear(256, 256, bias=True) for _ in range(3)], pseudo_grad_ops_rec, 
        [self.tc1_cdecay, self.tc1_vdecay, self.vth, self.grad_win])

        self.rec1 = RecurrentCUBALIFCell(nn.Identity() , nn.Linear(256, 256, bias=True),  pseudo_grad_ops_rec, 
        [self.r1_cdecay, self.r1_vdecay, self.grad_win, self.th_amp, self.th_decay, self.base_th])
        
        self.fc1 = FeedForwardCUBALIFCellDropout(nn.Linear(256, 256, bias=True), pseudo_grad_ops_rec, 
        [self.f1_cdecay, self.f1_vdecay, self.vth, self.grad_win])
        
        self.fc2 = nn.Linear(256, 2, bias=False)


    def forward(self, input_data, states):
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
        c1_state, c2_state, c3_state, tc1_state, r1_state, f1_state = \
            states[0], states[1], states[2], states[3], states[4], states[5]
        
        mask_tc = Bernoulli(
            torch.full_like(torch.zeros(batch_size, 256, device=torch.device("cuda")), 1 - dropout_tc)).sample() / (
                       1 - dropout_tc)

        mask = Bernoulli(
            torch.full_like(torch.zeros(batch_size, 256, device=torch.device("cuda")), 1 - dropout_rec)).sample() / (
                       1 - dropout_rec)

        mask_fc = Bernoulli(
            torch.full_like(torch.zeros(batch_size, 256, device=torch.device("cuda")), 1 - dropout_fc)).sample() / (
                       1 - dropout_fc)

        for step in range(self.spike_ts):
            input_spike = input_data[:, :, :, :, step]
            c1_spike, c1_state = self.conv1(input_spike, c1_state, )
            # ====== 插入CBAM（新增代码） ======
            c1_spike_reshaped = c1_spike.view(batch_size, 64, -1)  # 展平空间维度
            c1_spike = self.cbam_conv1(c1_spike_reshaped).view(batch_size, 64, 8, 9)  # 注意力加权+恢复形状
            # ===============================
            c2_spike, c2_state = self.conv2(c1_spike, c2_state)
            avg_pool_c2_spike = self.avg_pool(c2_spike)
            
            c3_spike, c3_state = self.conv3(avg_pool_c2_spike, c3_state)
            # ====== 插入CBAM（conv3后） ======
            c3_spike_reshaped = c3_spike.view(batch_size, 256, -1)  # [batch,256,1] → [batch,256,1]
            c3_spike_att = self.cbam_conv3(c3_spike_reshaped)
            c3_spike = c3_spike_att.view(batch_size, 256, 1, 1)
            # ===============================
            flat_c3_spike = c3_spike.view(batch_size, -1)
            
            temp_conv_spike_buffer.append(flat_c3_spike)
            
            if len(temp_conv_spike_buffer) > 3:
                temp_conv_spike_buffer.pop(0)

            tc1_spike, tc1_state = self.temp_conv1(temp_conv_spike_buffer, tc1_state, mask_tc, self.training)

            r1_spike, r1_state = self.rec1(tc1_spike, r1_state, mask, self.training)
            
            f1_spike, f1_state = self.fc1(r1_spike, f1_state, mask_fc, self.training)
            f2_output = self.fc2(f1_spike)
            output_spikes += [f2_output * self.ts_weights[step]]
        outputs = torch.stack(output_spikes).sum(dim=0) 
        return outputs

class WrapCUBASpikingCNN(nn.Module):
    def __init__(self, spike_ts, device, param_list, record_neuron=None):
        """
        :param spike_ts: spike timesteps
        :param device: device
        :param param_list: list of param for each neuron layer
        """
        super(WrapCUBASpikingCNN, self).__init__()
        self.device = device
        self.cdecay, self.vdecay, self.vth, self.grad_win, self.th_amp, self.th_decay, self.base_th = param_list
        self.snn = CUBASpikingCNN(spike_ts, param_list, record_neuron=record_neuron)

    def forward(self, input_data):
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
        r1_current = torch.zeros(batch_size, 256, device=self.device)
        r1_volt = torch.zeros(batch_size, 256, device=self.device)
        r1_spike = torch.zeros(batch_size, 256, device=self.device)
        r1_vth = torch.ones(batch_size, 256, device=self.device) * self.vth
        r1_state = (r1_spike, r1_current, r1_volt, r1_vth)
        f1_current = torch.zeros(batch_size, 256, device=self.device)
        f1_volt = torch.zeros(batch_size, 256, device=self.device)
        f1_spike = torch.zeros(batch_size, 256, device=self.device)
        f1_state = (f1_spike, f1_current, f1_volt)
        states = (c1_state, c2_state, c3_state, tc1_state, r1_state, f1_state)
        output = self.snn(input_data, states)

        return output
