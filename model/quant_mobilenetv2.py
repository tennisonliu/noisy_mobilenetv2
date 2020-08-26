import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from .quant_utils import _make_divisible
from .quant_layers import NoisyConv2d, NoisyLinear, ConvBNReLU, InvertedResidual, View

class QuantisedMobileNetV2(nn.Module):
    def __init__(self, q_a=0, q_w=0, dropout=0.0, num_classes=10, 
                 width_mult=1.0, inverted_residual_setting=None, round_nearest=8,
                 quant_three_sig=False, debug=False):
        '''
        Model declaration for quantise-able MobileNetV2

        q_a: number of bits to quantise layer input (default 0 => no quantisation)
        q_scale: value to scale upper value of quantised layer by (default 1 => no scaling)
        q_calculate_running: calclate running average of activations (default False)
        q_pctl: percentile of input/activation clipping used in quantisation (default 99.98)
        dropout: percentage of random zero-outs (default 0.0)
        '''
        super(QuantisedMobileNetV2, self).__init__()
        self.q_a = q_a
        self.q_w = q_w
        self.quant_three_sig = quant_three_sig
        self.debug = debug
        self.dropout = dropout
        self.arrays = []
        
        print('Model Initialisation:')
        print(f'Quantising activations to {self.q_a} bits')
        print(f'Quantising weights to {self.q_w} bits')
        print(f'Quantising to three-sigma range: {self.quant_three_sig}, debug enabled: {self.debug}')

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        pooling_kernel = 4

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                # [6, 24, 2, 2],
                [6, 24, 2, 1],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, q_a=self.q_a, q_w=self.q_w, 
                                stride=1, quant_three_sig=self.quant_three_sig, debug=self.debug)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride=stride, expand_ratio=t, 
                                      q_a=self.q_a, q_w=self.q_w, quant_three_sig=self.quant_three_sig, debug=self.debug))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, q_a=self.q_a, q_w=self.q_w, 
                                   kernel_size=1, quant_three_sig=self.quant_three_sig, debug=self.debug))
        self.features = nn.Sequential(*features)
        # self.drop1 = nn.Dropout(self.dropout)
        self.pool1 = nn.AvgPool2d(pooling_kernel)
        self.view1 = View()
        self.fc1 = NoisyLinear(self.last_channel, num_classes, bias=True, num_bits=self.q_a, num_bits_weight=self.q_w, 
                               quant_three_sig=self.quant_three_sig, debug=self.debug)
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
                if m.bias is not None:
                    nn.init.zeros_(m.bias) #TODO: the last FC layer should have bias

    def forward(self, x):
        # print('HERE')
        # network architecture
        # x = self.features(x, epoch, i)
        x = self.features(x)
        x = self.pool1(x)
        x = self.view1(x, x.size(0))
        # x = x.mean([2, 3])
        # dropout
        # if self.dropout > 0:
        #     x = self.drop1(x)

            # add dropout quantisation
            # x = quantise(x)
        # TODO: change the global argument params
        # if self.q_a > 0:
        #     x = self.quantise(x)
        # last fc
        x = self.fc1(x)

        return x