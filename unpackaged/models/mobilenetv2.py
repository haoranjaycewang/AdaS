"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace

class Block(nn.Module):
    #expand+depthwise+pointwise

    def __init__(self,in_planes, out_planes_1, out_planes_2, out_planes, stride, shortcut=False):
        self.in_planes=in_planes
        self.out_planes=out_planes
        self.shortcut=shortcut
        super(Block,self).__init__()
        self.stride=stride

        self.conv1=nn.Conv2d(in_planes, out_planes_1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1=nn.BatchNorm2d(out_planes_1)
        self.conv2=nn.Conv2d(out_planes_1, out_planes_2, kernel_size=3, stride=stride, padding=1, groups=out_planes_2, bias=False)
        self.bn2=nn.BatchNorm2d(out_planes_2)
        self.conv3=nn.Conv2d(out_planes_2, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3=nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride==1 and in_planes!=out_planes and shortcut!=False:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self,y):
        x = F.relu(self.bn1(self.conv1(y)))
        #print(x.shape, 'post conv1')
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.shape, 'post conv2')
        x = self.bn3(self.conv3(x))
        #print(x.shape, 'post conv3')
        #if self.shortcut!=nn.Sequential():
            #print(x.shape, 'out')
            #print(y.shape, 'in')
            #print(self.shortcut(y).shape, 'shortcut_in')
        x = x + self.shortcut(y) if (self.stride == 1) else x
        return x

class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        #self.index=[32, 32, 32, (16,'shortcut'), 16, 96, 96, (24,'shortcut'), 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, (96,'shortcut'), 96, 576, 576, 96, 576, 576, 96, 576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, (320,'shortcut'), 320, 1280]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ########################################StepLR Choices#####################################
        ######################################## 0% #############################################
        #self.index=[32, 32, 32, (16,'shortcut'), 16, 96, 96, (24,'shortcut'), 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, (96,'shortcut'), 96, 576, 576, 96, 576, 576, 96, 576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, (320,'shortcut'), 320, 1280]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ######################################## 20% ############################################
        #self.index_ORIGINAL=[26, 26, 26, (14,'shortcut'), 14, 78, 78, (20,'shortcut'), 20, 116, 116, 20, 116, 116, 28, 156, 156, 28, 156, 156, 28, 156, 156, 56, 310, 310, 54, 310, 310, 54, 310, 310, 54, 310, 310, (80,'shortcut'), 80, 464, 464, 82, 464, 464, 80, 466, 466, 134, 772, 772, 134, 774, 774, 132, 774, 774, (258,'shortcut'), 258, 1026]
        #self.index=[32, 26, 26, (14,'shortcut'), 14, 78, 78, (20,'shortcut'), 20, 116, 116, 20, 116, 116, 28, 156, 156, 28, 156, 156, 28, 156, 156, 56, 310, 310, 56, 310, 310, 56, 310, 310, 56, 310, 310, (80,'shortcut'), 80, 464, 464, 80, 464, 464, 80, 466, 466, 134, 772, 772, 134, 774, 774, 134, 774, 774, (258,'shortcut'), 258, 1026]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ######################################## 40% ############################################
        #self.index=[32, 20, 20, (10,'shortcut'), 12, 60, 60, (18,'shortcut'), 18, 90, 90, 18, 90, 90, 24, 120, 120, 24, 118, 118, 24, 120, 120, 46, 236, 236, 46, 236, 236, 46, 236, 236, 46, 238, 238, (66,'shortcut'), 64, 354, 354, 64, 354, 354, 64, 356, 356, 108, 586, 586, 108, 588, 588, 108, 588, 588, (196,'shortcut'), 194, 772]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ######################################## 60% ############################################
        #self.index=[32, 16, 16, (8,'shortcut'), 8, 42, 42, (14,'shortcut'), 14, 62, 62, 14, 62, 62, 18, 82, 82, 18, 82, 82, 18, 84, 84, 38, 162, 162, 38, 162, 162, 38, 162, 162, 38, 164, 164, (50,'shortcut'), 48, 242, 242, 48, 242, 242, 48, 246, 246, 82, 398, 398, 82, 404, 404, 82, 404, 404, (134,'shortcut'), 132, 518]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ######################################## 80% ############################################
        #self.index=[32, 10, 10, (6,'shortcut'), 6, 24, 24, (10,'shortcut'), 10, 34, 34, 10, 36, 36, 14, 46, 46, 14, 46, 46, 14, 48, 48, 28, 88, 88, 28, 88, 88, 28, 88, 88, 28, 90, 90, (36,'shortcut'), 32, 130, 130, 32, 132, 132, 32, 136, 136, 56, 210, 210, 56, 218, 218, 56, 218, 218, (70,'shortcut'), 70, 262]
        #self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        ######################################## 100% ###########################################
        self.index=[32, 4, 4, (2,'shortcut'), 4, 6, 6, (8,'shortcut'), 8, 8, 8, 8, 8, 8, 10, 10, 10, 10, 8, 8, 10, 12, 12, 20, 14, 14, 20, 14, 14, 20, 14, 14, 20, 18, 18, (20,'shortcut'), 16, 18, 18, 16, 20, 20, 16, 26, 26, 32, 24, 24, 32, 32, 32, 32, 32, 32, (8,'shortcut'), 8, 8]
        self.strides_and_short=[(1,True),(1,True),(1,False),(2,False),(1,False),(1,False),(2,False),(1,False),(1,False),(1,False),(1,True),(1,False),(1,False),(2,False),(1,False),(1,False),(1,True)]
        #########################################################################################

        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, self.index[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.index[0])

        #self.layers = self._make_layers(in_planes=32)
        self.layers=self._create_network(Block)

        self.conv2 = nn.Conv2d(self.index[len(self.index)-2], self.index[len(self.index)-1], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(self.index[len(self.index)-1])
        self.linear = nn.Linear(self.index[len(self.index)-1], num_classes)

    def _create_network(self,block):
        layers=[]
        input_res=32
        i=0
        stride_i=0
        while i<len(self.index)-4:
            if isinstance(self.index[i],int)!=True:
                i+=1
            else:
                if isinstance(self.index[i+3],tuple)==True:
                    layers.append(block(self.index[i],self.index[i+1],self.index[i+2],self.index[i+4],1,shortcut=True))
                    i+=4
                    stride_i+=1
                else:
                    input_res=input_res/2
                    stride = self.strides_and_short[stride_i][0]#if input_res>2 else 1
                    #print(stride, stride_i, 'stride choice')
                    #shortcut=self.strides_and_short[other_i][1]
                    layers.append(block(self.index[i],self.index[i+1],self.index[i+2],self.index[i+3],stride))
                    i+=3
                    stride_i+=1
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
'''
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)
'''

def test():
    net = MobileNetV2()
    ##print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    #g=make_dot(y)
    #g.view()
    #print('true')
    print(y.size())
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #Baseline: 2296922, 20%: 1552752, 40%: 947996, 60%: 493046, 80%: 181324, 100%: 16818.
    #print(pytorch_total_params)

#test()
