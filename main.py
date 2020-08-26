'''Train MobileNetV2 on CIFAR10'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from model import *
from misc_utils import progress_bar, train, test
import logging
import customlogger

# Presently not used
# logger = customlogger.setup_custom_logger('root')

def main():
    # Define training hyperparameters
    best_acc, best_epoch = 0, 0
    epochs = 350
    batch_size = 128
    learning_rate = 0.1
    weight_decay = 4e-5
    lr_schedule = [150, 250]
    lr_decay = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> Building model..')

    args = {'q_a': 0, 
            'q_w': 0, 
            'quant_three_sig': False,
            'debug': False}

    net = QuantisedMobileNetV2(**args)
    # net = MobileNetV2()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # from torchsummary import summary
    # print('Model Summary')
    # summary(net_init.cuda(), (3, 32, 32))
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=weight_decay)

    print('==> Training model..')
    for epoch in range(epochs):
        train(net, criterion, optimizer, epoch, trainloader, device, lr_decay, lr_schedule)
        if (res := test(net, criterion, testloader, device, save_best=True, epoch=epoch, best_results=[best_acc, best_epoch], save_model_path='best_net.pth')) is not None: [best_acc, best_epoch] = res

    print('==> Training complete..')
    print('Best Accuracy: %.4f at epoch %d' % (best_acc, best_epoch))

if __name__ == "__main__":
    main()