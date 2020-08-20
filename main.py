'''Train MobileNetV2 on CIFAR10'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
from models import *
from utils import progress_bar
import logging
import customlogger

# Presently not used
# logger = customlogger.setup_custom_logger('root')

def train(net, criterion, optimizer, epoch, trainloader, device, lr_decay, lr_schedule=[]):
    '''Train network for one epoch'''

    if epoch in lr_schedule:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_decay
        print('Updating LR')
    print('\nEpoch: %d and LR: %.4f' % (epoch, optimizer.param_groups[0]["lr"]))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(net, criterion, epoch, testloader, device, best_results):
    '''Evaluate network'''

    [best_acc, _] = best_results
    print('\nEvaluating...')
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        best_epoch = epoch
        return best_acc, best_epoch


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
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> Building model..')
    net = MobileNetV2()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                        momentum=0.9, weight_decay=weight_decay)

    print('==> Training model..')
    for epoch in range(epochs):
        train(net, criterion, optimizer, epoch, trainloader, device, lr_decay, lr_schedule)
        if (res := test(net, criterion, epoch, testloader, device, [best_acc, best_epoch])) is not None: [best_acc, best_epoch] = res
        
    print('==> Training complete..')
    print('Best Accuracy: %.4f at epoch %d' % (best_acc, best_epoch))

if __name__ == "__main__":
    main()