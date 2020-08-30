import torch
from torch import nn
from torch.autograd.function import InplaceFunction, Function
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import matplotlib.pyplot as plt

class NoisyConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,       bias=False, num_bits=0, num_bits_weight=0, quant_three_sig=False, debug=False):
        super(NoisyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        # self.noise = noise
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight
        self.quant_three_sig = quant_three_sig
        self.debug = debug
        if self.num_bits > 0:
            # self.quantise_input = QuantMeasure(self.num_bits, three_sig=True)
            self.quantise_input = UniformQuantise.apply
        if self.num_bits_weight > 0:
            # min_value, max_value set here tells QuantMeasure to quantise weights
            # self.quantise_weights = QuantMeasure(self.num_bits_weight, three_sig=True)
            self.quantise_weights = UniformQuantise.apply
    
    def forward(self, input):
        weight = self.weight
        bias = self.bias

        # print('IN the CONV2D forward function, {}'.format(self.training))
        # only quantise during inference
        if self.num_bits > 0 and not self.training:
            input_scale = self.input_scale
            qinput = self.quantise_input(input, input_scale, self.num_bits, self.quant_three_sig, False, self.debug) # num_bits=8, three_sig=False, inplace=False
            # debugging
            if self.debug:
              print('Debugging activations before and after quantisation')
              print(f'Input shape: {input.size()}')
              ax1 = plt.subplot(2, 1, 1)
              ax1.hist(input.cpu().reshape(-1), bins=20)
              ax2 = plt.subplot(2, 1, 2)
              ax2.hist(qinput.cpu().reshape(-1), bins=20)
              # plt.title("Activations before (top) and after (bottom) quantisation")
              plt.show()
        else:
            qinput = input
        # only quantise during inference
        if self.num_bits_weight > 0 and not self.training:
            weight = self.quantise_weights(self.weight, self.num_bits_weight, self.quant_three_sig, False, self.debug)
            # debugging
            if self.debug:
              print('Debugging weights before and after quantisation')
              print(f'Weight shape: {self.weight.size()}')
              ax = plt.subplot(2, 1, 1)
              ax.hist(self.weight.cpu().reshape(-1), bins=20)
              ax = plt.subplot(2, 1, 2)
              ax.hist(weight.cpu().reshape(-1), bins=20)
              plt.show()

        # if not self.training: print('Done Debugging')
        output = F.conv2d(qinput, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return output

class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, num_bits=0, num_bits_weight=0,
                 quant_three_sig=False, debug=False):
        super(NoisyLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight
        self.quant_three_sig = quant_three_sig
        self.debug = debug
        if num_bits > 0:
            # self.quantise_input = QuantMeasure(self.num_bits, three_sig=True)
            self.quantise_input = UniformQuantise.apply
            self.quantise_output = UniformQuantise.apply
        if num_bits_weight > 0:
            # self.quantise_weights = QuantMeasure(self.num_bits_weight, three_sig=True)
            self.quantise_weights = UniformQuantise.apply

    def forward(self, input):
        weight = self.weight
        bias = self.bias

        # only quantise during inference
        if self.num_bits > 0 and not self.training:
            input_scale = self.input_scale
            qinput = self.quantise_input(input, input_scale, self.num_bits, self.quant_three_sig, False, self.debug) # num_bits=8, three_sig=False, inplace=False
            # debugging
            if self.debug:
              print('Debugging activations before and after quantisation')
              print(f'Input shape: {input.size()}')
              ax = plt.subplot(2, 1, 1)
              ax.hist(input.cpu().view(-1))
              ax = plt.subplot(2, 1, 2)
              ax.hist(qinput.cpu().view(-1))
              plt.show()
        else:
            qinput = input
        # only quantise during inference
        if self.num_bits_weight > 0 and not self.training:
            weight = self.quantise_weights(self.weight, self.num_bits_weight, self.quant_three_sig, False, self.debug)
            if self.debug:
              print('Debugging weights before and after quantisation')
              print(f'Weight shape: {self.weight.size()}')
              ax = plt.subplot(2, 1, 1)
              ax.hist(self.weight.cpu().view(-1))
              ax = plt.subplot(2, 1, 2)
              ax.hist(weight.cpu().view(-1))
              plt.show()

        output = F.linear(qinput, weight, bias)
        # Quantise output
        # if self.num_bits > 0 and not self.training:        
        #     output_scale = self.output_scale
        #     output = self.quantise_output(output, output_scale, self.num_bits, self.quant_three_sig, False, self.debug)

        return output

# Conv + BatchNorm + ReLU block
class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, q_a=0, q_w=0, kernel_size=3, stride=1, groups=1, 
                 track_running_stats=False, quant_three_sig=False, debug=False):
        super(ConvBNReLU, self).__init__()
        self.q_a = q_a
        self.q_w = q_w
        self.track_running_stats = track_running_stats
        self.quant_three_sig = quant_three_sig
        self.debug = debug
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, self.padding, groups=groups, bias=False)
        self.conv = NoisyConv2d(in_planes, out_planes, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, groups=self.groups, bias=False, num_bits=self.q_a, num_bits_weight=self.q_w, quant_three_sig=self.quant_three_sig, debug=self.debug)
        self.bn = nn.BatchNorm2d(out_planes, track_running_stats=self.track_running_stats)
        # self.relu = nn.ReLU6(inplace=True)
        self.relu = nn.ReLU()
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.groups = groups
        # if self.q_a > 0:
        #     self.quantise = QuantMeasure(self.q_a, scale=self.q_scale, calculate_running=self.q_calculate_running, pctl=self.q_pctl / 100)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, q_a, q_w, 
                track_running_stats=False, quant_three_sig=False, debug=False):
        '''
        Create inverted residual module
        inp: layer input
        oup: layer output
        stride: stride for Conv2d modules
        expand ratio: expansion factor for layer in inverted residual block
        '''
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.q_a = q_a
        self.q_w = q_w
        self.track_running_stats = track_running_stats
        self.quant_three_sig = quant_three_sig
        self.debug = debug
        # residual connection
        self.use_res_connect = self.stride == 1 and inp == oup

        assert self.stride in [1, 2]
        self.hidden_dim = int(round(inp * self.expand_ratio))

        self.conv1 = ConvBNReLU(inp, self.hidden_dim, self.q_a, self.q_w, kernel_size=1,
                                track_running_stats = self.track_running_stats, quant_three_sig=self.quant_three_sig, 
                                debug=self.debug)
        self.conv2 = ConvBNReLU(self.hidden_dim, self.hidden_dim, self.q_a, self.q_w, 
                                stride=self.stride, groups=self.hidden_dim, track_running_stats=self.track_running_stats,
                                quant_three_sig=self.quant_three_sig, debug=self.debug)
        # self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.conv3 = NoisyConv2d(self.hidden_dim, oup,  kernel_size=1 , stride=1, padding=0, bias=False, 
                                 num_bits=self.q_a, num_bits_weight=self.q_w,
                                 quant_three_sig=self.quant_three_sig, debug=self.debug)
        self.bn = nn.BatchNorm2d(oup, track_running_stats=self.track_running_stats)

        # TODO: remove global params
        # if self.q_a > 0:
            # self.quantize1 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl / 100, debug=args.debug_quant)
            # self.quantize2 = QuantMeasure(args.q_a, stochastic=args.stochastic, scale=args.q_scale, calculate_running=args.calculate_running, pctl=args.pctl / 100, debug=args.debug_quant)
            # self.quantise = QuantMeasure(self.q_a, scale=self.q_scale, calculate_running=self.calculate_running, pctl=self.pctl / 100)

    def forward(self, x):
        input = x
        if self.expand_ratio != 1:
            x = self.conv1(x)
        x = self.conv2(x)
        # TODO
        # if self.q_a > 0:
        #     x = self.quantise(x)
        x = self.conv3(x)
        x = self.bn(x)

        if self.use_res_connect:
            return x + input
        else:
            return x

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
    
    def forward(self, x, shape):
        return x.view(shape, -1)

'''
class UniformQuantise(InplaceFunction):
    """modified from https://github.com/eladhoffer/quantized.pytorch/blob/master/models/modules/quantize.py"""

    @staticmethod
    def forward(ctx, input, num_bits=8, three_sig=False, inplace=False, debug=False):
        # pctl is not used for the moment
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        # TODO: need to set this for straight-through estimation
        ctx.save_for_backward(input)

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        min_value = torch.min(output)
        max_value = torch.max(output)

        # if using 3-sigma range
        if three_sig:
            if debug:
                print("Using 3-sigma range")
            mean = torch.mean(output)
            std = torch.std(output)
            min_value = mean - (3 * std)
            max_value = mean + (3 * std)

        ctx.min_value = min_value
        ctx.max_value = max_value

        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1.
        scale = (max_value - min_value) / (qmax - qmin)
        scale = max(scale, 1e-6)  # TODO figure out how to set this robustly! causes nans

        if debug:
            print(f'qmin: {qmin}, qmax: {qmax}, raw_min: {min_value.item()}, raw_max:{max_value.item()}')
        with torch.no_grad():
            # output.add_(-min_value).div_(scale).add_(qmin)
            output.div_(scale)
            output.clamp_(qmin, qmax).round_()  # quantize

            # TODO: figure out the purpose behind dequantisation?
            # output.add_(-qmin).mul_(scale).add_(min_value)  # dequantize

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Saturated Straight Through Estimator
        input, = ctx.saved_tensors
        # Should we clone the grad_output???
        grad_output[input > ctx.max_value] = 0
        grad_output[input < ctx.min_value] = 0
        # grad_input = grad_output
        return grad_output, None, None, None, None, None, None
'''

class UniformQuantise(InplaceFunction):
    """modified from https://github.com/eladhoffer/quantized.pytorch/blob/master/models/modules/quantize.py"""

    @staticmethod
    def forward(ctx, input, scale_factor, num_bits=8, three_sig=False, inplace=False, debug=False):
        # pctl is not used for the moment
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        # TODO: need to set this for straight-through estimation
        ctx.save_for_backward(input)

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        qmin = - 2. ** (num_bits - 1)
        qmax = 2. ** (num_bits - 1) - 1.

        with torch.no_grad():
            # output.add_(-min_value).div_(scale).add_(qmin)
            output = (scale_factor * output)
            # output.div_(scale_factor)
            output.clamp_(qmin, qmax).round_()  # quantize

            # TODO: figure out the purpose behind dequantisation?
            # output.add_(-qmin).mul_(scale).add_(min_value)  # dequantize

        if debug:
            min_value = torch.min(input)
            max_value = torch.max(input)
            print(f'raw_min: {min_value.item()}, raw_max:{max_value.item()}')
            print(f'Quantisation range: {qmin}, {qmax}')
            print(f'Output Scale factor: {scale_factor}')
            print(f'q_min: {torch.min(output).item()}, q_max:{torch.max(output).item()}')

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Saturated Straight Through Estimator
        input, = ctx.saved_tensors
        # Should we clone the grad_output???
        grad_output[input > ctx.max_value] = 0
        grad_output[input < ctx.min_value] = 0
        # grad_input = grad_output
        return grad_output, None, None, None, None, None, None