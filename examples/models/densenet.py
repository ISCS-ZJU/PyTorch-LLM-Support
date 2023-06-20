'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
max_mem = 0
first_time = 0
def do_atm(x, MANUAL_PAGEOUT, PREFETCH_NEED):
    global max_mem
    prev_mem = max_mem
    max_mem = max(max_mem, torch.cuda.memory_allocated('cuda:0'))
    if (MANUAL_PAGEOUT):
        x.pageout_manual()
        if (PREFETCH_NEED):
            x.need_prefech()
def reset_atm():
    global max_mem
    max_mem = 0

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, MANUAL_PAGEOUT_  = False, PREFETCH_NEED_ = False):
        super(Bottleneck, self).__init__()
        self.MANUAL_PAGEOUT  = MANUAL_PAGEOUT_
        self.PREFETCH_NEED   = PREFETCH_NEED_

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)


    def forward(self, x):
        global first_time
        out1 = self.bn1(x)
        out2 = F.relu(out1)
        out3 = self.conv1(out2)
        out4 = self.bn2(out3)
        out5 = F.relu(out4)
        out6 = self.conv2(out5)
        out7 = torch.cat([out6,x], 1)
        return out7


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, MANUAL_PAGEOUT_  = False, PREFETCH_NEED_ = False):
        super(Transition, self).__init__()
        self.MANUAL_PAGEOUT  = MANUAL_PAGEOUT_
        self.PREFETCH_NEED   = PREFETCH_NEED_

        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out0 = self.bn(x)
        out00 = F.relu(out0)
        out1 = self.conv(out00)
        out2 = F.avg_pool2d(out1, 2)
        return out2


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, MANUAL_PAGEOUT_  = False, PREFETCH_NEED_ = False):
        super(DenseNet, self).__init__()
        self.MANUAL_PAGEOUT  = MANUAL_PAGEOUT_
        self.PREFETCH_NEED   = PREFETCH_NEED_

        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock, MANUAL_PAGEOUT_, PREFETCH_NEED_):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, MANUAL_PAGEOUT_, PREFETCH_NEED_))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        reset_atm()
        out1 = self.conv1(x)
        out = self.trans1(self.dense1(out1))
        print("Alloced: %.5f MB" % (torch.cuda.memory_allocated('cuda:0') / 1024 / 1024))
        out = self.trans2(self.dense2(out))
        print("Alloced: %.5f MB" % (torch.cuda.memory_allocated('cuda:0') / 1024 / 1024))
        out = self.trans3(self.dense3(out))
        print("Alloced: %.5f MB" % (torch.cuda.memory_allocated('cuda:0') / 1024 / 1024))
        out = self.dense4(out)
        print("Alloced: %.5f MB" % (torch.cuda.memory_allocated('cuda:0') / 1024 / 1024))
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        print(out.size())
        out = self.linear(out)
        return out
    
class DenseNet_IN(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, MANUAL_PAGEOUT_  = False, PREFETCH_NEED_ = False):
        super(DenseNet_IN, self).__init__()
        self.MANUAL_PAGEOUT  = MANUAL_PAGEOUT_
        self.PREFETCH_NEED   = PREFETCH_NEED_

        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(num_planes)
        self.relu1 =  nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0], self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes, self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1], False, False)
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes, self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2], False, False)
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes, False, False)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3], False, False)
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock, MANUAL_PAGEOUT_, PREFETCH_NEED_):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, MANUAL_PAGEOUT_, PREFETCH_NEED_))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        reset_atm()
        out = self.conv1(x)
        do_atm(x, self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        out1 = self.norm1(out)
        do_atm(out, self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        out2 = self.relu1(out1)
        out = self.pool1(out2)
        do_atm(out2, self.MANUAL_PAGEOUT, self.PREFETCH_NEED)
        out = self.trans1(self.dense1(out))
        print("Alloced: %.5f MB" % (torch.cuda.memory_allocated('cuda:0') / 1024 / 1024))
        sys.stdout.flush()
        out = self.trans2(self.dense2(out))
        print("Alloced: %.5f MB" % (torch.cuda.memory_allocated('cuda:0') / 1024 / 1024))
        sys.stdout.flush()
        out = self.trans3(self.dense3(out))
        print("Alloced: %.5f MB" % (torch.cuda.memory_allocated('cuda:0') / 1024 / 1024))
        sys.stdout.flush()
        out = self.dense4(out)
        print("Alloced: %.5f MB" % (torch.cuda.memory_allocated('cuda:0') / 1024 / 1024))
        sys.stdout.flush()
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), (1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def DenseNet121(MANUAL_PAGEOUT  = False, PREFETCH_NEED = False, num_classes=10):
    print(MANUAL_PAGEOUT, PREFETCH_NEED)
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_classes=num_classes, MANUAL_PAGEOUT_=MANUAL_PAGEOUT, PREFETCH_NEED_=PREFETCH_NEED)

def DenseNet121_IN(MANUAL_PAGEOUT  = False, PREFETCH_NEED = False, num_classes=1000):
    print(MANUAL_PAGEOUT, PREFETCH_NEED)
    return DenseNet_IN(Bottleneck, [6,12,24,16], growth_rate=32, num_classes=num_classes, MANUAL_PAGEOUT_=MANUAL_PAGEOUT, PREFETCH_NEED_=PREFETCH_NEED)

def DenseNet169(MANUAL_PAGEOUT  = False, PREFETCH_NEED = False):
    print(MANUAL_PAGEOUT, PREFETCH_NEED)
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, MANUAL_PAGEOUT_=MANUAL_PAGEOUT, PREFETCH_NEED_=PREFETCH_NEED)

def DenseNet201(MANUAL_PAGEOUT  = False, PREFETCH_NEED = False):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, MANUAL_PAGEOUT_=MANUAL_PAGEOUT, PREFETCH_NEED_=PREFETCH_NEED)

def DenseNet161(MANUAL_PAGEOUT  = False, PREFETCH_NEED = False):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, MANUAL_PAGEOUT_=MANUAL_PAGEOUT, PREFETCH_NEED_=PREFETCH_NEED)

def densenet_cifar(MANUAL_PAGEOUT  = False, PREFETCH_NEED = False):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, MANUAL_PAGEOUT_=MANUAL_PAGEOUT, PREFETCH_NEED_=PREFETCH_NEED)

