import torch
from torch import nn
from torch.autograd.function import InplaceFunction, Function
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

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

def _make_divisible(v, divisor, min_value=None):
    '''
    Ensure that all layers have a channel number that is divisible by 8
    '''
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
