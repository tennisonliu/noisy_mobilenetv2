import torch
from torch import nn
from torch.autograd.function import InplaceFunction, Function
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from .quant_utils import QuantMeasure
import matplotlib.pyplot as plt

class NoisyConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 num_bits=0, num_bits_weight=0, q_scale=1, q_calculate_running=False, q_pctl=99.98):
        super(NoisyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.noise = noise
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight
        self.q_scale = q_scale
        self.q_calculate_running = q_calculate_running
        self.q_pctl = q_pctl
        if self.num_bits > 0:
            self.quantise_input = QuantMeasure(self.num_bits, scale=self.q_scale, calculate_running=self.q_calculate_running, pctl=self.q_pctl/100)
        if self.num_bits_weight > 0:
            # min_value, max_value set here tells QuantMeasure to quantise weights
            self.quantise_weights = QuantMeasure(self.num_bits_weight, min_value=-1.0, max_value=1.0)
    
    def forward(self, input):
        weight = self.weight
        bias = self.bias
        # only quantise during inference
        if self.num_bits > 0 and not self.training:
            qinput = self.quantise_input(input)
            # # debugging
            # print('Debugging activations before and after quantisation')
            # print(f'Input shape: {input.size()}')
            # ax1 = plt.subplot(2, 1, 1)
            # ax1.hist(input.cpu().reshape(-1), bins=20)
            # ax2 = plt.subplot(2, 1, 2)
            # ax2.hist(qinput.cpu().reshape(-1), bins=20)
            # plt.title("Activations before (top) and after (bottom) quantisation")
            # plt.savefig(f'debug/activations.pdf')
        else:
            qinput = input
        # only quantise during inference
        if self.num_bits_weight > 0 and not self.training:
            weight = self.quantise_weights(self.weight)
            # # debugging
            # print('Debugging weights before and after quantisation')
            # print(f'Weight shape: {self.weight.size()}')
            # plt.title("Weights before (top) and after (bottom) quantisation")
            # ax = plt.subplot(2, 1, 1)
            # ax.hist(self.weight.cpu().reshape(-1), bins=20)
            # ax = plt.subplot(2, 1, 2)
            # ax.hist(weight.cpu().reshape(-1), bins=20)
            # plt.savefig(f'debug/weights.pdf')

        # if not self.training: print('Done Debugging')
        output = F.conv2d(qinput, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return output

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, num_bits=0, num_bits_weight=0, q_scale=1, q_calculate_running=False, q_pctl=99.98):
        super(NoisyLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight
        self.q_scale = q_scale
        self.q_calculate_running = q_calculate_running
        self.q_pctl = q_pctl
        if num_bits > 0:
            self.quantise_input = QuantMeasure(self.num_bits, scale=self.q_scale, calculate_running=self.q_calculate_running, pctl=self.q_pctl/100)
        if num_bits_weight > 0:
            self.quantise_weights = QuantMeasure(self.num_bits_weight, min_value=-1.0, max_value=1.0, )

    def forward(self, input):
        weight = self.weight
        bias = self.bias

        # only quantise during inference
        if self.num_bits > 0 and not self.training:
            qinput = self.quantise_input(input)
            # debugging
            # if epoch==0 and i==0:
            #     print('Debugging activations before and after quantisation')
            #     ax = plt.subplot(2, 1, 1)
            #     ax.hist(input.cpu().view(-1))
            #     ax = plt.subplot(2, 1, 2)
            #     ax.hist(qinput.cpu().view(-1))
            #     plt.title("Activations before (top) and after (bottom) quantisation")
            #     plt.savefig(f'debug/activations_{i}.pdf')
        else:
            qinput = input
        # only quantise during inference
        if self.num_bits_weight > 0 and not self.training:
            weight = self.quantise_weights(self.weight)
            # if epoch==0 and i==0:
            #     print('Debugging weights before and after quantisation')
            #     ax = plt.subplot(2, 1, 1)
            #     ax.hist(self.weight.cpu().view(-1))
            #     ax = plt.subplot(2, 1, 2)
            #     ax.hist(weight.cpu().view(-1))
            #     plt.title("Weights before (top) and after (bottom) quantisation")
            #     plt.savefig(f'debug/activations_{i}.pdf')

        output = F.linear(qinput, weight, bias)
        return output