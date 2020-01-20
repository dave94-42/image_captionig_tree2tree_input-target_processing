import torch.nn as nn
from torchvision import models

from tree import *


class MyAlexNet(nn.Module):

    "myAlexNet(i.e. alexNet with only features cutted to 3rd conv layer)"

    original_model = models.alexnet(pretrained=True)

    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
        # stop at 3rd conv layer
        *list(self.original_model.features.children())[:8]
        )

    def forward(self, x):
        x=self.features(x)
        return x



class MyAlexNetMap(torch.nn.Module):

    "my alex for map i.e. same as myAlex but convlotuins go from one channel to one"

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,1, kernel_size=11, stride=4,padding=2 ,bias=False)
        self.conv2 = torch.nn.Conv2d(1,1, kernel_size=5, padding=2,bias=False)
        self.conv3 = torch.nn.Conv2d(1,1, kernel_size=3, padding=1, bias=False)
        self.avg1 = torch.nn.AvgPool2d(kernel_size=3, stride=2)
        self.avg2 = torch.nn.AvgPool2d(kernel_size=3, stride=2)

        self.conv1.weight.data.copy_(  torch.ones([1,1,11,11])/121 )
        self.conv2.weight.data.copy_( torch.ones([1,1,5,5]) /25 )
        self.conv3.weight.data.copy_( torch.ones([1,1,3,3]) /9 )


    def forward(self, x):
        shape = x.shape
        x = self.conv1(x)
        x = self.avg1(x)

        shape = x.shape
        x = self.conv2(x)
        x = self.avg2(x)

        shape = x.shape
        x = self.conv3(x)
        return x