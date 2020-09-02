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

def main(quant_bits, q_calculate_running, profile_activations, profile_epochs, model_savename):
    # Define training hyperparameters
    best_acc, best_epoch = 0, 0
    epochs = 350
    batch_size = 128
    learning_rate = 0.1
    weight_decay = 4e-5
    lr_schedule = [150, 250]
    lr_decay = 0.1
    track_running_stats = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainloader, testloader = prepare_data('CIFAR10', train_batchsize=128)

    criterion = nn.CrossEntropyLoss()

    print('==> Building model..')
    args = {'q_a': quant_bits, 
            'q_w': quant_bits, 
            'quant_three_sig': False,
            'track_running_stats': track_running_stats,
            'q_calculate_running': q_calculate_running,
            'debug': False}
    net = QuantisedMobileNetV2(**args)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=weight_decay)

    print('==> Training model..')
    for epoch in range(epochs):
        gathering_stats = profile_activations and epoch <= profile_epochs-1
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.gathering_stats = gathering_stats
        print(f'Gathering Stats? {gathering_stats}')

        train(net, criterion, optimizer, epoch, trainloader, device, lr_decay, lr_schedule)
        if (res := test(net, criterion, testloader, device, 
                        save_best=True, epoch=epoch, best_results=[best_acc, best_epoch], 
                        save_model_path=model_savename)) is not None: [best_acc, best_epoch] = res

    print('==> Training complete..')
    print('Best Accuracy: {} at epoch {}'.format(best_acc, best_epoch))

    
if __name__ == "__main__":
    quant_bits = 8
    q_calculate_running = True
    profile_activations = True
    profile_epochs = 5
    model_savename = 'qat_net.pth'
    main(quant_bits, q_calculate_running, profile_activations, profile_epochs, model_savename)