'''Post Training Quantisation of MobileNetv2'''
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
from model import *
from typing import Tuple
from misc_utils import copy_model, test
import matplotlib.pyplot as plt

def quantised_weights(weights: torch.Tensor, num_bits=8, debug=False) -> Tuple[torch.Tensor, float]:
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
        ax = plt.subplot(2, 1, 1)    
        ax.hist(weights.cpu().view(-1))
        ax = plt.subplot(2, 1, 2)
        ax.hist(quant_weights.cpu().view(-1))
        plt.show()
        print(f'qmin: {qmin}, qmax: {qmax}, scale_factor: {scale}, raw_min: {min_value.item()}, raw_max:{max_value.item()}')

    return quant_weights, scale

def quantise_layer_weights(model: nn.Module, device, num_bits=8, debug=False):
    count = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            print(layer)
            count += 1
            q_layer_data, scale = quantised_weights(layer.weight.data, num_bits, debug)
            q_layer_data = q_layer_data.to(device)

            layer.weight.data = q_layer_data
            layer.weight.scale = scale

            if (q_layer_data < -(2.**(num_bits-1))).any() or (q_layer_data > 2.**(num_bits-1)-1).any():
                raise Exception("quantised weights of {} layer include values out of bounds for an 8-bit signed integer".format(layer.__class__.__name__))
            if (q_layer_data != q_layer_data.round()).any():
                raise Exception("quantised weights of {} layer include non-integer values".format(layer.__class__.__name__))

    print(f'Weights of {count} layers quantised')
    assert count==54

def main(saved_model_path, quant_bits):
    # Initialise Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    print('==> Building model..')
    args = {'q_a': 0, 
            'q_w': 0, 
            'quant_three_sig': False,
            'track_running_stas': False,
            'debug': False}
    net = QuantisedMobileNetV2(**args)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    print(net)

    # Copy weights into net
    if os.path.isfile(saved_model_path):
        print("=> loading checkpoint '{}'".format(saved_model_path))
        checkpoint = torch.load(saved_model_path)
        net.load_state_dict(checkpoint['net'])
    else:
        print("=> no checkpoint found at '{}'".format(saved_model_path))

    # Evaluate on test set
    print('Evaluating 32FP MobileNetV2')
    best_acc, best_epoch = test(net, criterion, testloader, device, save_best=False)
    print('[Raw MobileNetv2] - Best Accuracy: {} at epoch {}'.format(best_acc, best_epoch))

    # Quantise layer weights and evaluate
    print('Evaluating MobileNet with {} bits quantised weights'.format(quant_bits))
    quant_net = copy_model(net)
    quantise_layer_weights(quant_net, device=device, num_bits=quant_bits, debug=True)
    best_acc, best_epoch = test(quant_net, criterion, testloader, device, save_best=False)
    print('[Quantised Weights MobileNetv2] - Best Accuracy: {} at epoch {}'.format(best_acc, best_epoch))
    
    '''
    # Quantise layer activations
    print('Evalating MobileNetV2 with %d quantised weights and activations' % quant_bits)
    # Create model
    args = {'q_a': quant_bits, 
            'q_w': 0, 
            'quant_three_sig': False,
            'debug': False}
    quant_net_2 = QuantisedMobileNetV2(**args)
    quant_net_2 = quant_net_2.to(device)
    if device == 'cuda':
        quant_net_2 = torch.nn.DataParallel(quant_net_2)
        cudnn.benchmark = True

    # Profile activations using raw model (i.e. raw weights) and training data
    net_temp = copy_model(net)
    register_activation_profiling_hooks(net_temp)
    test(net_temp, trainloader, save_best=False, max_samples=500)
    net_temp.profile_activations = False
    # quantise activations
    quantise_activations(net_temp, num_bits=quant_bits, debug=True)

    # Copy weights, weight scales across
    quant_net_2 = copy_model(net, quant_net_2)

    # Instantiate model again, this time with quantisation parameters, copy weights across

    print('[Quantised Weights + Activations MobileNetv2] - best Accuracy: %.4f at epoch %d' % (best_acc, best_epoch))
    '''

if __name__ == "__main__":
    saved_model_path = './checkpoint/best_net.pth'
    quant_bits = 8
    main(saved_model_path, quant_bits)