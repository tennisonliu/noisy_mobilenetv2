'''Quantised MobileNetV2 Architecture'''
from typing import Tuple
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch

def quantised_weights(weights: torch.Tensor, num_bits) -> Tuple[torch.Tensor, float]:
    '''
    Post Training Quantiation (Symmetric Quantisation - Signed Integers)
    Quantise weights so that all values are integers between [-2**n, 2**n-1]
    Presently, only use total range to scale the input values by

    Parameters:
    weights: unquantised weights

    Returns:
    (Tensor, float): (weights in quantised form, scaling factor)
    '''
    max_value = torch.max(weights)
    min_value = torch.min(weights)
    qmin = float(-2**num_bits)
    qmax = float(2**num_bits-1)

    scale = (qmax - qmin)/(max_value - min_value)
    scale = max(scale, 1e-6)

    # weights quantisation
    weights = torch.clamp((weights * scale).round(), min=qmin, max=qmax)

    return weights, scale

def quantise_layer_weights(model: nn.Module):
    '''
    Given a model, quantise layer weights
    Presently only quantises Conv2d and Linear layers
    '''
    for layer in model.children():
        print(layer)
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            q_layer_data, scale = quantised_weights(layer.weight.data)
            q_layer_data = q_layer_data.to(device)

            layer.weight.data = q_layer_data
            layer.weight.scale = scale

            if (q_layer_data < -128).any() or (q_layer_data > 127).any():
                raise Exception("Quantized weights of {} layer include values out of bounds for an 8-bit signed integer".format(layer.__class__.__name__))
            if (q_layer_data != q_layer_data.round()).any():
                raise Exception("Quantized weights of {} layer include non-integer values".format(layer.__class__.__name__))


def register_activation_profiling_hooks(model: nn.Module):
    '''
    Add forward hooks for Conv2D adn Linear layers in network

    Parameters:
    model: network to profile
    '''
    model.profile_activations = True
    def save_activation(previous_layer, layer, name, previous_layer_name, layer_no, mod, inp, out):
      # print(f"Forward step through {name}, previous layer {previous_layer_name}")
      if model.profile_activations:
        if layer_no == 1:
          '''first layer register input activations'''
          model.input_activations = np.append(model.input_activations, inp[0].cpu().view(-1))
          # print(f"{name}, First Layer")
        elif layer_no == total_layer_no - 1:
            '''final layer register penultimate and final layer activations'''
          previous_layer.activations = np.append(previous_layer.activations, inp[0].cpu().view(-1))
          layer.activations = np.append(layer.activations, out[0].cpu().view(-1))
          # print(f"{name}, Last Layer")
        else:
          previous_layer.activations = np.append(previous_layer.activations, inp[0].cpu().view(-1))

    # initialise input and layer activations
    model.input_activations = np.empty(0)
    for name, layer in model.named_modules():
      layer.activations = np.empty(0)

    total_layer_no = len(list(model.named_modules()))
    previous_layer, previous_layer_name = None, None
    # create forward hooks for conv2d and linear layers
    for layer_no, (name, layer) in enumerate(model.named_modules()):
      if name is not '' and (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
        print(f"registering hooks for layer: {name}...")
        layer.register_forward_hook(partial(save_activation, previous_layer, layer, name, previous_layer_name, layer_no))
        previous_layer, previous_layer_name = layer , name

class QuantBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(QuantBlock, self).__init__()
        self.stride = stride

        planes = expansion * in_planes

        # expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # depthwise convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # pointwise convolution
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        # shortcut
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


'''Quantised MobileNetV2 Architecture'''
class QuantisedMobileNetV2(nn.Module):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, net_with_weights_quantised: nn.Module, num_bits: int):
        super(QuantisedMobileNetv2, self).__init__()
        
        net_init = copy_model(net_with_weights_quantised)

        for layer in net_init.children():
            print(layer)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                def pre_hook(l, x):
                    x = x[0]
                    if (x < -128).any() or (x > 127).any():
                        raise Exception("Input to {} layer is out of bounds for an 8-bit signed integer".format(l.__class__.__name__))
                    if (x != x.round()).any():
                        raise Exception("Input to {} layer has non-integer values".format(l.__class__.__name__))

                layer.register_forward_pre_hook(pre_hook)
        
        self.num_bits = num_bits
        self.input_activations = net_with_weights_quantised.input_activations
        self.input_scale = QuantisedMobileNetV2.quantise_initial_input(self.input_activations, self.num_bits)

        preceding_layer_scales = []
        for layer in net_init.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.output_scale = QuantisedMobileNetV2.quantise_activations(layer.activations, layer.weight.scale, self.input_scale, preceding_layer_scales, num_bits)
                preceding_layer_scales.append((layer.weight.scale, layer.output_scale))
    
    @staticmethod
    def quantise_initial_input(pixels: np.ndarray, num_bits: int) -> float:
        '''
        Calculate scaling factor for input to first layer

        Parameters:
        pixels: values of all pixels which were part of the input image during training

        Returns:
        float: scaling factor for input
        '''
        max_value = np.max(pixels)
        min_value = np.min(pixels)
        qmin = float(-2**num_bits)
        qmax = float(2**num_bits-1)
        scale = (qmax - qmin)/(max_value - min_value)
        scale = max(scale, 1e-6)
        return scale

    
    def quantise_activations(activations: np.ndarray, n_w: float, n_initial_input: float, ns: List[Tuple[float, float]], num_bits: int) -> float:
        '''
        Calculate scaling factor to multiple output of a layer

        Parameters:
        activations: activation values output by layer during training
        n_w: scale by which weights of layer were multiplied
        n_initla_input: scale by which initial input to the neural network was multiplied
        ns: list of tuples, each tuple represents weight scale, output scale of every preceding layer

        Returns:
        float: scaling factor for the layer output
        '''
        quant_activations = activations * float(n_w) * n_initial_input
        for preceding_layer_activations in ns:
            quant_activations *= float(preceding_layer_activations[0])
            quant_activations *= float(preceding_layer_activations[1])
        max_value = np.max(quant_activations)
        min_value = np.min(quant_activations)
        qmin = float(-2**num_bits)
        qmax = float(2**num_bits-1)
        scale = (qmax - qmin)/(max_value - min_value)
        scale = max(scale, 1e-6)
        return scale
    
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(QuantBlock(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)


    def quant_forward(x):
        out = F.relu(net_init.bn1(net_init.conv1(x)))
        out = net_init.layers(out)
        out = F.relu(net_init.bn2(net_init.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = net_init.linear(out)
        return out

