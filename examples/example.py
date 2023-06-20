import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import torchvision
from torchvision.models.alexnet import alexnet
# from torchvision.models.alexnet import alexnet
import torchvision.transforms as transforms
import sys
import os
import time
import random
# python3 cifar10.py  --tensorboard import SummaryWriter
from models import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--workers_num', type = int, default = 4)
parser.add_argument('--batch_size', type = int, default = 512)
parser.add_argument('--epochs_num', type = int, default = 120)
parser.add_argument('--target_acc', type = int, default = 100)
parser.add_argument('--print', type = int, default = 0)

parser.add_argument('--lms', dest='userEnabledLMS', action='store_true')
parser.add_argument('--no-lms', dest='userEnabledLMS', action='store_false')
parser.set_defaults(userEnabledLMS=True)
parser.add_argument('--lms-limit', default=0, type=int, help='limit (in MB)')

parser.add_argument('--swap', action='store_true')

args = parser.parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.enabled = False

setup_seed(2022)

print_flag = int(args.print)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

print("\033[0;31;40mTarget acc is: \033[0m",args.target_acc, " %")
train_end = False

train_tags = ['Train-Acc','Train-Loss']


test_tags = ['Test-Acc', 'Test-Loss']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print("Epoch is ",str(args.epochs_num))
print("Batch size is ",str(args.batch_size))
# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # Can be saved
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./dataset/Cifar-10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.workers_num))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
best_acc = 0



# Model
print('==> Building model..')
net = DenseNet121(args.swap, args.swap)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
print(args.swap)
# Training
def train(epoch):
    max_batch = 2
    print('Epoch {}/{}'.format(epoch + 1, str(args.epochs_num)))
    print('-' * 10)
    start_time = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    end = time.time()
    tot_time = 0.0
    if args.swap:
        torch.cuda.start_evict_env()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.swap:
            torch.cuda.prefetch_init()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if args.swap:
            torch.cuda.before_prefetch_wait_all()
        print('------------------------------training batch %d' % batch_idx)
        sys.stdout.flush()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        batch_time = time.time() - end

        print("batch time : %.3f" % batch_time)
        end = time.time()

        with open('acc_pattern/acc_pattern_mnist_batch%02d.csv' % batch_idx, 'w') as file_out:
            file_out.write(torch.cuda.get_storageimpl_profile())
        with open('debug_log/debug_log_%02d.log' % batch_idx, 'w') as file_out:
            file_out.write(torch.cuda.get_debug_atm())
        # print(torch.cuda.get_debug_atm())
        torch.cuda.clear_storageimpl_profile()
        torch.cuda.clear_debug_atm()
        print('------------------------------trained batch %d' % batch_idx)
        print('----------------------------------------------')
        print('----------------------------------------------')
        print('----------------------------------------------')
        sys.stdout.flush()
        
        if (batch_idx > 0):
            tot_time += batch_time
        if batch_idx == max_batch:
            print(tot_time / max_batch)
            break
    if args.swap:
        torch.cuda.stop_evict_env()
    end_time = time.time()
    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, end_time-start_time))

    acc = 100.*correct/total

train_begin = time.time()
for epoch in range(start_epoch, start_epoch+int(args.epochs_num)):
    train(epoch)
