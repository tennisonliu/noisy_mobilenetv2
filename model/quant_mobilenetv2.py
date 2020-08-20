import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from quant import QuantMeasure
from quant_layers import NoisyConv2d, NoisyLinear

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

# Conv + BatchNorm + ReLU block
class ConvBNReLU(nn.Module):
    def __init__(self, in_planes, out_planes, q_a=0, q_w=0, q_scale=1, q_calculate_running=False, q_pctl=99.98, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.q_a = q_a
        self.q_w = q_w
        self.q_scale = q_scale
        self.q_calculate_running = q_calculate_running
        self.q_pctl = q_pctl
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, self.padding, groups=groups, bias=False)
        self.conv = NoisyConv2d(in_planes, out_planes, kernel_size=self.kernel_size, stride=self.stride, self.padding, groups=self.groups, bias=False, num_bits=self.q_a, num_bits_weight=self.q_w, q_scale=self.q_scale, q_calculate_running=self.q_calculate_running, q_pctl=self.q_pctl)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6(inplace=True)
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
    def __init__(self, inp, oup, stride, expand_ratio, q_a, q_w, q_scale, q_calculate_running, q_pctl):
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
        self.q_scale = q_scale
        self.q_calculate_running = q_calculate_running
        self.q_pctl = q_pctl
        # residual connection
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv1 = ConvBNReLU(inp, hidden_dim, self.q_a, self.q_w, self.q_scale, self.q_calculate_running, self.q_pctl, kernel_size=1)
        self.conv2 = ConvBNReLU(hidden_dim, hidden_dim, self.q_a, self.q_w, self.q_scale, self.q_calculate_running, self.q_pctl, stride=self.stride, groups=hidden_dim)
        # self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.conv3 = NoisyConv2d(hidden_dim, oup,  kernel_size=1 , stride=1, padding=0, bias=False, num_bits=self.q_a, num_bits_weights=self.q_w)
        self.bn = nn.BatchNorm2d(oup)

        assert self.stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))

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

class QuantisedMobileNetV2(nn.Module):
    def __init__(self, q_a=0, q_w=0, q_scale=1, q_calculate_running=False, q_pctl=99.98, dropout=0.2, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        '''
        Model declaration for quantise-able MobileNetV2

        q_a: number of bits to quantise layer input (default 0 => no quantisation)
        q_scale: value to scale upper value of quantised layer by (default 1 => no scaling)
        q_calculate_running: calclate running average of activations (default False)
        q_pctl: percentile of input/activation clipping used in quantisation (default 99.98)
        dropout: percentage of random zero-outs (default 0.0)
        '''
        super(MobileNetV2, self).__init__()
        self.q_a = q_a
        self.q_w = q_w
        self.q_scale = q_scale
        self.q_calculate_running = q_calculate_running
        self.q_pctl = q_pctl
        self.dropout = dropout
        self.arrays = []
        
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, q_a=self.q_a, q_w=self.q_w, q_scale=self.q_scale, q_calculate_running=self.q_calculate_running, q_pctl=self.q_pctl, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride=stride, expand_ratio=t, q_a=self.q_a, q_w=self.q_w, q_scale=self.q_scale, q_calculate_running=self.q_calculate_running, q_pctl=self.q_pctl))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, q_a=self.q_a, q_w=self.q_w, q_scale=self.q_scale, q_calculate_running=self.q_calculate_running, q_pctl=self.q_pctl, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.drop1 = nn.Dropout(self.dropout)
        self.fc1 = NoisyLinear(self.last_channel, num_classes, num_bits=self.q_a, num_bits_weights=self.q_w, q_scale=self.q_scale, q_calculate_running=self.q_calculate_running, q_pctl=self.q_pctl)
        # quantisation for dropout, if used
        # self.quantise = QuantMeasure(self.q_a, pctl=self.q_pctl, max_value=)

        # q_a => number of bits to quantise layer input 
        # TODO: change this if not using global argument
        # if self.q_a > 0:
        #     self.quantise = QuantMeasure(self.q_a, scale=self.q_scale, calculate_running=self.q_calculate_running, pctl=self.q_pctl / 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, epoch=0, i=0, acc=0.0):
        # network architecture
        x = self.features(x)
        x = x.mean([2, 3])
        # dropout
        if self.dropout > 0:
            x = self.drop1(x)

            # add dropout quantisation
            # x = quantise(x)
        # TODO: change the global argument params
        # if self.q_a > 0:
        #     x = self.quantise(x)
        # last fc
        x = self.fc1(x)

        return x