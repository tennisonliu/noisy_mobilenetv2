'''Post Training Quantisation of MobileNetv2'''
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
from model import *
from model.quant_utils import _quantise_layer_weights, _register_activation_profiling_hooks, _get_layer_quant_factor, _calc_quant_scale
from misc_utils import copy_model, train, test, prepare_data
import matplotlib.pyplot as plt
from copy import deepcopy

def main(saved_model_path, quant_bits, pctl_range):
    # Initialise Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader = prepare_data('CIFAR10', train_batchsize=128)

    criterion = nn.CrossEntropyLoss()

    print('==> Building model..')
    args = {'q_a': 0, 
            'q_w': 0, 
            'quant_three_sig': False,
            'track_running_stats': False,
            'debug': False}
    net = QuantisedMobileNetV2(**args)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    print(net)

    # Copy weights into net
    if os.path.isfile(saved_model_path):
        print("==> loading checkpoint '{}'".format(saved_model_path))
        checkpoint = torch.load(saved_model_path)
        net.load_state_dict(checkpoint['net'])
    else:
        print("==> no checkpoint found at '{}'".format(saved_model_path))

    # Evaluate on test set
    print('\nEvaluating 32FP MobileNetV2')
    best_acc, best_epoch = test(net, criterion, testloader, device, save_best=False)
    print('\n[Raw MobileNetv2] - Best Accuracy: {} at epoch {}'.format(best_acc, best_epoch))

    # Quantise layer weights and evaluate
    print('\nEvaluating MobileNet with {} bits quantised weights'.format(quant_bits))
    quant_net = copy_model(net)
    _quantise_layer_weights(quant_net, device=device, num_bits=quant_bits, debug=True)
    best_acc, best_epoch = test(quant_net, criterion, testloader, device, save_best=False)
    print('\n[Quantised Weights MobileNetv2] - Best Accuracy: {} at epoch {}'.format(best_acc, best_epoch))
    
    # Quantise layer activations
    print('\nEvalating MobileNetV2 with {} bits quantised weights and activations'.format(quant_bits))
    # Profile output activations witht train data
    print('\n==> Profiling activations..')
    _register_activation_profiling_hooks(quant_net)
    print('\n==> Profiling activations using train data')
    test(quant_net, criterion, trainloader, device, save_best=False, max_batches=2)
    quant_net.profile_activations = False
    # Calculate quantisation scale factor based on activation profiles
    _calc_quant_scale(quant_net, num_bits=quant_bits, pctl_range=pctl_range, debug=False)

    # Create net with quantisation during forward steps. 
    # Note: no q_w since quantised weights have already been copied over
    args = {'q_a': quant_bits, 
            'q_w': 0, 
            'quant_three_sig': False,
            'track_running_stats': False,
            'debug': False}
    quant_net_2 = QuantisedMobileNetV2(**args)
    quant_net_2 = quant_net_2.to(device)
    if device == 'cuda':
        quant_net_2 = torch.nn.DataParallel(quant_net_2)
        cudnn.benchmark = True
    print('\n==> Creating network with quantisation function')

    # Copy quantised weights from quant_net
    quant_net_2.load_state_dict(quant_net.state_dict())

    # Copy activation scale factor from quant_net
    layer_count = 0
    for layer_init2, layer_init in zip(quant_net_2.modules(), quant_net.modules()):
        if isinstance(layer_init2, nn.Conv2d):
            layer_count += 1
            if layer_count != 2:
                layer_init2.input_scale = deepcopy(layer_init.input_scale)
        if isinstance(layer_init2, nn.Linear):
            layer_count += 1
            layer_init2.input_scale = deepcopy(layer_init.input_scale)
            # layer_init2.output_scale = deepcopy(layer_init.output_scale)
    assert layer_count==54
    
    best_acc, best_epoch = test(quant_net_2, criterion, testloader, device, save_best=False)
    print('\n[Quantised Weights + Activations MobileNetv2] - Best Accuracy: {} at epoch {}'.format(best_acc, best_epoch))
    
if __name__ == "__main__":
    saved_model_path = './checkpoint/best_net.pth'
    quant_bits = 4
    pctl_range = 99.7 # three sigma range
    main(saved_model_path, quant_bits, pctl_range)