import torch
from torch import nn
from torch.autograd.function import InplaceFunction, Function
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

class UniformQuantize(InplaceFunction):
    """modified from https://github.com/eladhoffer/quantized.pytorch/blob/master/models/modules/quantize.py"""

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, inplace=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.save_for_backward(input)

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        qmin = 0.
        qmax = 2. ** num_bits - 1.
        scale = (max_value - min_value) / (qmax - qmin)
        scale = max(scale, 1e-6)  # TODO figure out how to set this robustly! causes nans

        with torch.no_grad():
            output.add_(-min_value).div_(scale).add_(qmin)
            output.clamp_(qmin, qmax).round_()  # quantize
            output.add_(-qmin).mul_(scale).add_(min_value)  # dequantize

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

class QuantMeasure(nn.Module):
    '''
    Calculate_running indicates if we want to calculate the given percentile of signals to use as a max_value for quantization range
    if True, we will calculate pctl for several batches (only on training set), and use the average as a running_max, which will became max_value
    if False we will either use self.max_value (if given), or self.running_max (previously calculated)
    Currently, calculate_running param is set in the training code, and NOT passed as an argument - TODO need to fix that
    If using dropout, during training the activations are divided by 1-p. Multiply calculate_running by 1-p during test.
    '''

    def __init__(self, num_bits=8, momentum=0.0, min_value=0., max_value=0., scale=1,
                 calculate_running=False, pctl=90., inplace=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros([]))
        self.momentum = momentum
        self.num_bits = num_bits
        self.inplace = inplace
        self.max_value = max_value
        self.min_value = min_value
        self.scale = scale
        self.calculate_running = calculate_running
        self.running_list = []
        self.pctl = pctl

    def forward(self, input):
        with torch.no_grad():
            min_value = self.min_value
            if self.calculate_running:
                # during training
                if self.min_value < 0:  
                    # quantising weights (since ReLU is always positive)
                    pctl_pos, _ = torch.kthvalue(input[input > 0].flatten(), int(input[input > 0].numel() * self.pctl / 100.))
                    pctl_neg, _ = torch.kthvalue(torch.abs(input[input < 0]).flatten(), int(input[input < 0].numel() * self.pctl / 100.))
                    self.running_min = -pctl_neg
                    self.running_max = pctl_pos
                    self.calculate_running = False
                    min_value = self.running_min.item()
                    max_value = self.running_max.item()
                else:
                    # quantising inputs
                    if 32 in list(input.shape):  #TODO: CHANGE first layer input (CIFAR-10) needs more precision (at least 6 bits)
                        # if input
                        if self.num_bits == 4:
                            pctl = torch.tensor(0.92)
                        else:
                            pctl = torch.tensor(1.0)
                    else:
                        # if hidden layers
                        pctl, _ = torch.kthvalue(input.flatten(), int(input.numel() * self.pctl / 100.)) # return kth smallest integer

                    max_value = input.max().item()
                    self.running_list.append(pctl)  # self.running_max

            else:
                # during testing
                if self.min_value < 0 and self.running_min < 0:
                    min_value = self.running_min.item()
                    max_value = self.running_max.item()
                elif self.max_value > 0:
                    max_value = self.max_value
                elif self.running_max.item() > 0:
                    max_value = self.running_max.item()
                else:
                    # print('\n\nSetting max_value to input.max\nrunning_max is ', self.running_max.item())
                    max_value = input.max().item()

                if False and max_value > 1:
                    max_value = max_value * self.scale

            # if self.training:
            #     stoch = self.stochastic
            # else:
            #     stoch = 0

        # return UniformQuantize().apply(input, self.num_bits, min_value, max_value, stoch, self.inplace, False)
        return UniformQuantize().apply(input, self.num_bits, min_value, max_value, self.inplace)