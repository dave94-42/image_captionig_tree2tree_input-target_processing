import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from tree import *


class MyInception(nn.Module):

    "myAlexNet(i.e. alexNet with only features cutted to 3rd conv layer)"

    original_model =models.inception_v3(pretrained=True,aux_logits=False,transform_input=False)

    def __init__(self):
        super(MyInception, self).__init__()
        self.Conv2d_1a_3x3 = self.original_model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = self.original_model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = self.original_model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = self.original_model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = self.original_model.Conv2d_4a_3x3

    def forward(self, x):
        x=self.Conv2d_1a_3x3(x)
        x=self.Conv2d_2a_3x3(x)
        x=self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x=self.Conv2d_3b_1x1(x)
        x=self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        return x



class MyInceptionMap(torch.nn.Module):

    "my alex for map i.e. same as myAlex but convlotuins go from one channel to one"

    def __init__(self):
        super(MyInceptionMap, self).__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(1, 1, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(1,1, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(1, 1, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(1, 1, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(1, 1, kernel_size=3)

    def forward(self, x):
        x=self.Conv2d_1a_3x3(x)
        x=self.Conv2d_2a_3x3(x)
        x=self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x=self.Conv2d_3b_1x1(x)
        x=self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class OriginalInception(nn.Module):

    original_model =models.inception_v3(pretrained=True,aux_logits=False,transform_input=True)

    def __init__(self):
        super(OriginalInception,self).__init__()
        self.Conv2d_1a_3x3 = self.original_model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = self.original_model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = self.original_model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = self.original_model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = self.original_model.Conv2d_4a_3x3
        self.Mixed_5b = self.original_model.Mixed_5b
        self.Mixed_5c = self.original_model.Mixed_5c
        self.Mixed_5d = self.original_model.Mixed_5d
        self.Mixed_6a = self.original_model.Mixed_6a
        self.Mixed_6b = self.original_model.Mixed_6b
        self.Mixed_6c = self.original_model.Mixed_6c
        self.Mixed_6d = self.original_model.Mixed_6d
        self.Mixed_6e = self.original_model.Mixed_6e
        self.Mixed_7a = self.original_model.Mixed_7a
        self.Mixed_7b = self.original_model.Mixed_7b
        self.Mixed_7c = self.original_model.Mixed_7c

    def forward(self, x):
        #transform input
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        #forward
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        #x = x.view(2048*8*8)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.shape[0],2048)
        return x