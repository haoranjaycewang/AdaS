"""
MIT License

Copyright (c) 2017 liukuang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software withx restriction, including withx limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHx WARRANTY OF ANY KIND, Ex PRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
x OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, x iangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arx iv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace

from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models

import torch.onnx


class BasicBlock(nn.Module):

    def __init__(self, in_planes, intermediate_planes, out_planes,stride=1):
        self.in_planes=in_planes
        self.intermediate_planes=intermediate_planes
        self.out_planes=out_planes

        super(BasicBlock,self).__init__()
        '''if in_planes!=intermediate_planes:
            #print('shortcut_needed')
            stride=2
        else:
            stride=stride'''
        self.conv1=nn.Conv2d(
                in_planes,
                intermediate_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
        )
        self.bn1=nn.BatchNorm2d(intermediate_planes)
        self.conv2=nn.Conv2d(
                intermediate_planes,
                out_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
        )
        self.bn2=nn.BatchNorm2d(out_planes)
        self.relu=nn.ReLU()
        self.shortcut=nn.Sequential()
        if stride!=1 or in_planes!=out_planes:
            #print('shortcut_made')
            self.shortcut=nn.Sequential(
                nn.Conv2d(
                        in_planes,
                        out_planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False
                ),
                nn.BatchNorm2d(out_planes),
                #nn.ReLU()
            )

    def forward(self,y):
        x = self.conv1(y)

        #print(x.shape,'post conv1 block')
        x = self.bn1(x)
        x = F.relu(x)

        x = self.bn2(self.conv2(x))

        #print(x.shape,'post conv2 block')
        #if self.shortcut!=nn.Sequential():
            #print('shortcut_made')
        #print(self.shortcut)
        #print(x.shape)
        #print(y.shape)
        #print(self.shortcut(y).shape)

        x += self.shortcut(y)
        #print(x.shape,'post conv3 block')
        x = F.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, image_channels,num_classes=10):
        super(ResNet, self).__init__()

        ################################################################################## SGD StepLR ##################################################################################

        ####################### O% ########################

        self.superblock1_indexes=[64, 5, 64, 6, 64, 7, 64]
        self.superblock2_indexes=[8, 128, 9, 128, 10, 128, 11, 128]
        self.superblock3_indexes=[12, 256, 13, 256, 14, 256, 15, 256, 16, 256, 17, 256]
        self.superblock4_indexes=[18, 512, 19, 512, 20, 512]
        '''
        self.superblock1_indexes=[64, 64, 64, 64, 64, 64, 64]
        self.superblock2_indexes=[128, 128, 128, 128, 128, 128, 128, 128]
        self.superblock3_indexes=[256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]
        self.superblock4_indexes=[512, 512, 512, 512, 512, 512]
        '''
        ####################### 20% #######################
        #self.index=[52, 54, 54, 54, 54, 54, 54, 108, 104, 108, 106, 108, 106, 108, 106, 214, 206, 212, 210, 212, 210, 212, 210, 212, 210, 210, 208, 414, 412, 412, 412, 412, 412]
        ####################### 40% #######################
        #self.index=[40, 42, 44, 44, 44, 44, 44, 86, 80, 86, 84, 88, 84, 88, 84, 170, 158, 170, 164, 168, 164, 170, 164, 168, 162, 166, 160, 314, 312, 312, 310, 314, 312]
        ####################### 60% #######################
        #self.index=[32, 32, 34, 34, 34, 32, 34, 66, 56, 66, 62, 68, 62, 68, 62, 128, 108, 126, 118, 126, 118, 126, 118, 124, 116, 120, 112, 216, 210, 212, 210, 214, 210]
        ####################### 80% #######################
        #self.index=[32, 22, 24, 24, 24, 22, 26, 44, 32, 44, 40, 48, 40, 48, 40, 84, 60, 82, 72, 82, 72, 84, 70, 80, 70, 74, 62, 118, 110, 112, 108, 114, 110]
        ####################### 100% #######################
        #self.index=[32, 10, 14, 12, 14, 12, 16, 24, 8, 24, 18, 28, 18, 26, 18, 42, 10, 40, 26, 38, 26, 40, 24, 36, 22, 30, 14, 20, 10, 12, 8, 16, 10]

        ##################################################################################    AdaS    ##################################################################################

        ####################### O% ########################
        #self.index=[64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        ####################### 20% #######################
        #self.index=[52, 54, 54, 54, 54, 54, 54, 108, 104, 108, 106, 108, 106, 106, 106, 214, 208, 212, 210, 212, 210, 212, 210, 212, 210, 212, 208, 418, 414, 416, 412, 414, 412]
        ####################### 40% #######################
        #self.index=[42, 44, 44, 44, 44, 44, 44, 86, 80, 86, 84, 86, 84, 86, 84, 170, 158, 170, 166, 170, 164, 168, 164, 168, 164, 168, 160, 326, 318, 322, 310, 316, 310]
        ####################### 60% #######################
        #self.index=[32, 32, 34, 34, 34, 34, 34, 66, 56, 66, 64, 66, 62, 64, 62, 128, 110, 126, 120, 126, 118, 124, 118, 122, 118, 124, 114, 232, 220, 226, 210, 216, 210]
        ####################### 80% #######################
        #self.index=[32, 22, 24, 22, 22, 22, 24, 46, 32, 44, 42, 46, 42, 44, 40, 84, 60, 82, 76, 84, 72, 80, 72, 78, 72, 78, 66, 140, 122, 132, 110, 118, 108]
        ####################### 100% #######################
        #self.index=[32, 12, 14, 12, 12, 12, 14, 24, 8, 24, 20, 24, 20, 22, 18, 42, 12, 38, 30, 40, 26, 36, 26, 34, 26, 34, 18, 46, 24, 36, 10, 20, 8]

        self.index=self.superblock1_indexes+self.superblock2_indexes+self.superblock3_indexes+self.superblock4_indexes

        self.num_classes=num_classes
        self.conv1 = nn.Conv2d(image_channels, self.index [0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.index[0])
        self.network=self._create_network(block)
        self.linear=nn.Linear(self.index[len(self.index )-1],num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu=nn.ReLU()

    def _create_network(self,block):
        output_size=56
        layers=[]
        layers.append(block(self.index[0],self.index[1],self.index[2],stride=1))
        for i in range(2,len(self.index)-2,2):
            #print(self.index [i],self.index [i+1],self.index [i+2],'for loop ',i)
            if (self.index[i]!=self.index[i+2]):
                stride=2
                output_size=int(output_size/2)
            else:
                stride=1
        #    if i==len(self.index)-4:
            #    self.linear=nn.Linear(self.index[len(self.index)-2],self.num_classes)
            layers.append(block(self.index[i],self.index[i+1],self.index[i+2],stride=stride))
        #    #print(i, 'i')
        #print(len(self.index),'len index')
        return nn.Sequential(*layers)

    '''
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.ex pansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.ex pansion
        return nn.Sequential(*layers)'''

    def forward(self, y):
        #print(self.index )
        x = self.conv1(y)
        #print(x.shape, 'conv1')
        x = self.bn1(x)
        #print(x.shape, 'bn1')
        x = F.relu(x)
        #print(x.shape, 'relu')
        #x = self.maxpool(x)
        ##print(x.shape, 'max pool')
        x = self.network(x)
        #print(x.shape, 'post bunch of blocks')
        x = self.avgpool(x)
        #print(x.shape, 'post avgpool')
        x = x.view(x.size(0), -1)
        #print(x.shape, 'post reshaping')
        x = self.linear(x)
        #print(x.shape, 'post fc')
        return x


def ResNet18(num_classes: int = 10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes = 10):
    return ResNet(BasicBlock, 3, num_classes=10)


def ResNet50(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes: int = 10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

def test():
    #writer = SummaryWriter('runs/resnet34_1')
    net = ResNet34()
    #print(net)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    g=make_dot(y)
    #g.view()
    torch.save(net.state_dict(),'temp_resnet.onnx')
    dummy_input = Variable(torch.randn(4, 3, 32, 32))
    torch.onnx.export(net, dummy_input, "model.onnx")
    #print(convCount)

#test()
