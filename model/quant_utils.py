import torch
from torch import nn
from torch.autograd.function import InplaceFunction, Function
import random
import numpy as np
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from typing import Tuple, List
import collections
from functools import partial

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

def _quantised_weights(weights: torch.Tensor, num_bits=8, debug=False) -> Tuple[torch.Tensor, float]:
    '''
    quantise the weights so that all values are integers between -128 and 127.
    You may want to use the total range, 3-sigma range, or some other range when
    deciding just what factors to scale the float32 values by.

    Parameters:
    weights (Tensor): The unquantised weights

    Returns:
    (Tensor, float): A tuple with the following elements:
                        * The weights in quantised form, where every value is an integer between -128 and 127.
                          The "dtype" will still be "float", but the values themselves should all be integers.
                        * The scaling factor that your weights were multiplied by.
                          This value does not need to be an 8-bit integer.
    '''

    max_value = torch.max(weights)
    min_value = torch.min(weights)

    qmin = - 2. ** (num_bits - 1)
    qmax = 2. ** (num_bits - 1) - 1.
    # scale = (max_value - min_value) / (qmax - qmin)
    # scale = max(scale, 1e-6)  # TODO figure out how to set this robustly! causes nans
    scale = (qmax - qmin)/(max_value - min_value)
    scale = max(scale, 1e-6)

    # quant_weights = weights
    # quant_weights.div_(scale)
    # quant_weights.clamp_(qmin, qmax).round_()
    quant_weights = (weights*scale)
    quant_weights.clamp_(qmin, qmax).round_()

    if debug:
        # ax = plt.subplot(2, 1, 1)    
        # ax.hist(weights.cpu().view(-1))
        # ax = plt.subplot(2, 1, 2)
        # ax.hist(quant_weights.cpu().view(-1))
        # plt.show()
        print(f'qmin: {qmin}, qmax: {qmax}, scale_factor: {scale}, raw_min: {min_value.item()}, raw_max:{max_value.item()}')

    return quant_weights, scale

def _quantise_layer_weights(model: nn.Module, device, num_bits=8, debug=False):
    count = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            print(layer)
            count += 1
            q_layer_data, scale = _quantised_weights(layer.weight.data, num_bits, debug)
            q_layer_data = q_layer_data.to(device)

            layer.weight.data = q_layer_data
            layer.weight.scale = scale

            if (q_layer_data < -(2.**(num_bits-1))).any() or (q_layer_data > 2.**(num_bits-1)-1).any():
                raise Exception("quantised weights of {} layer include values out of bounds for an 8-bit signed integer".format(layer.__class__.__name__))
            if (q_layer_data != q_layer_data.round()).any():
                raise Exception("quantised weights of {} layer include non-integer values".format(layer.__class__.__name__))

    print(f'Weights of {count} layers quantised')
    assert count==54

def _register_activation_profiling_hooks(model: nn.Module):
    '''
    Profile activations at the output of Conv+BN+ReLU or Conv+BN blocks
    '''
    model.profile_activations = True
    layer_count = 0
    def save_activation(layer, layer_no, mod, inp, out):
      # print(f"\n{layer}\n{layer_no}")
      if model.profile_activations:
        if layer_no == total_layer_no:
          '''final layer register penultimate and final layer activations'''
          layer.input_activations = np.append(layer.input_activations, inp[0].cpu().view(-1))
          layer.output_activations = np.append(layer.output_activations, out[0].cpu().view(-1))
          # print(f"{layer}, Last Layer")
        else:
          layer.input_activations = np.append(layer.input_activations, inp[0].cpu().view(-1))
          # print(f"{layer}, Layer in the Middle :)")

    # initialise input and layer activations
    # model.input_activations = np.empty(0)
    # for name, layer in model.named_modules():
    #   layer.activations = np.empty(0)

    total_layer_no = len([(name, layer) for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)])
    print(f'Total number of layers with quantisable activations {total_layer_no}')
    # create forward hooks for conv2d and linear layers
    for layer in model.modules():
      if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
        layer.input_activations = np.empty(0)
        layer_count += 1
        if layer_count == total_layer_no:
          layer.output_activations = np.empty(0)
        # There is no expansion layer in the first inverted residual block in MobileNetv2
        if layer_count != 2:
          print(f"registering hooks for layer {layer_count}: {layer}...")
          # print(f"with previous layer {previous_layer}\n")
          layer.register_forward_hook(partial(save_activation, layer, layer_count))
        else:
          print(f"no hooks registered for layer 2, which is the expansion layer for the first inverted residual bottleneck (skipped)")
    print(f'Registered activation hooks for {layer_count} layers')
    assert layer_count==54

def _get_layer_quant_factor(activations: np.ndarray, num_bits: int, ns: List[Tuple[float, float]], debug=False) -> float:
    '''
    Calculate a scaling factor to multiply the input of a layer by.

    Parameters:
    activations (ndarray): The values of all the pixels which have been output by this layer during training
    n_w (float): The scale by which the weights of this layer were multiplied as part of the "quantize_weights" function you wrote earlier
    n_initial_input (float): The scale by which the initial input to the neural network was multiplied
    ns ([(float, float)]): A list of tuples, where each tuple represents the "weight scale" and "output scale" (in that order) for every preceding layer

    Returns:
    float: A scaling factor that the layer output should be multiplied by before being fed into the first layer.
            This value does not need to be an 8-bit integer.
    '''

    quant_activations = activations
    '''
    quant_activations = activations * float(n_w) * n_initial_input
    for preceding_layer_activations in ns:
        print(preceding_layer_activations)
        # quant_activations *= float(preceding_layer_activations[0]) 
        quant_activations *= float(preceding_layer_activations[1])
    '''
    max_value = np.max(quant_activations)
    min_value = np.min(quant_activations)
    qmin = -2. ** (num_bits - 1)
    qmax = 2. ** (num_bits -1) - 1
    scale = (qmax - qmin)/(max_value - min_value)
    if debug:
      ax = plt.subplot(2, 1, 1)
      ax.hist(quant_activations)
      ax = plt.subplot(2, 1, 2)
      ax.hist(quant_activations*scale)
      plt.show()
    return scale

def _calc_quant_scale(model: nn.Module, num_bits, debug=False):

    preceding_layer_scales = []
    for layer_no, layer in enumerate(model.modules()):
      if isinstance(layer, nn.Conv2d):
        print(f'Layer: {layer_no}, {layer}')
        if not np.any(layer.input_activations):
          print(f'No input activations registered for layer {layer_no}')
          pass
        else:
          # layer.input_scale = get_layer_quant_factor(layer.input_activations, num_bits, layer.weight.scale, model.input_scale, preceding_layer_scales, debug=debug)
          layer.input_scale = _get_layer_quant_factor(layer.input_activations, num_bits, preceding_layer_scales, debug=debug)
          print(f'Scaling factor for layer input: {layer.input_scale}')
          preceding_layer_scales.append((layer.weight.scale, layer.input_scale))
      if isinstance(layer, nn.Linear):
        print(f'Layer: {layer_no}, {layer}')
        layer.input_scale = _get_layer_quant_factor(layer.input_activations, num_bits, preceding_layer_scales, debug=debug)
        preceding_layer_scales.append((layer.weight.scale, layer.input_scale))
        print(f'Scaling factor for layer input: {layer.input_scale}')
        layer.output_scale = _get_layer_quant_factor(layer.output_activations, num_bits, preceding_layer_scales, debug=debug)
        print(f'Scaling factor for layer output: {layer.output_scale}')