import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math
import time

__all__ = ['resnet50']

class ParamFree(nn.Module):
    def __init__(self,channel):
        super(ParamFree, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(2)
        self.gap4 = nn.AdaptiveAvgPool2d(4)
        self.sig = nn.Sigmoid()
        self.learn = Parameter(torch.zeros(3))
        self.batch_norm = nn.BatchNorm2d(channel)

    def forward(self, x):
        b, c, h, w = x.size()
        gap = self.gap(x)
        gap2 = self.gap2(x)
        gap4 = self.gap4(x)
        gapALL = (gap*self.learn[0]+gap4*self.learn[2]+F.interpolate(gap2, scale_factor=2)*self.learn[1])/3
        gapALL = F.interpolate(gapALL, size=(h, w))
        gapALL = self.batch_norm(gapALL)
        gapALL = self.sig(gapALL)

        return x*gapALL


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, modeltype=1, auto=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.falserelu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.modeltype = modeltype

        self.pf = ParamFree(planes * 4)

        if self.modeltype:
            self.gninner = nn.GroupNorm(32, planes, affine=True)
            self.gn4 = nn.GroupNorm(32, inplanes, affine=True)  # self.gn4 = nn.GroupNorm(32,planes * 4, affine=False)
            self.bn4 = nn.BatchNorm2d(planes * 4)
            self.avgpoolinner = nn.AvgPool2d(kernel_size=stride, stride=stride,
                                             padding=0)  # self.avgpool = nn.AvgPool2d(kernel_size = 3, stride=1, padding=1)
            if self.stride == 2:
                self.avgpoolinner = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        residual_inden = x
        x = self.falserelu(x)
        residual = x
        out = self.conv1(x)  # out = self.conv1(x) be careful, x or out
        out = self.bn1(out)
        out = self.relu(out)

        residual_inner = out
        out = self.conv2(out)
        out = self.bn2(out)
        if self.modeltype:
            residual_inner = self.gninner(residual_inner)
            # print(residual_inner.size())
            residual_inner = self.avgpoolinner(residual_inner) # residual_inner = F.avg_pool2d(residual_inner, kernel_size=3, stride=self.stride, padding=1) #
            # print(residual_inner.size())
        out += residual_inner
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.pf(out)
        if self.modeltype:
            residual = self.gn4(residual) + residual_inden
        if self.downsample is not None:
            residual = self.downsample(residual)  # residual not x # residual = self.downsample(x)

        out += residual

        # out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=365, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def demo():
    from torchstat import stat
    net = resnet50(num_classes=365)
    y = net(torch.randn(1, 3, 224, 224))
    print(y.size())
    stat(net, (3, 224, 224))

demo()
