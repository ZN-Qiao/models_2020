import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
# from .cbam import *
# from .bam import *
import pdb

__all__ = ['resnet50']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, modeltype=0, auto=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.modeltype = modeltype
        self.gn1 = nn.GroupNorm(32,inplanes)
        self.gn2 = nn.GroupNorm(32,planes)
        if self.modeltype:
            #self.gn1 = nn.GroupNorm(32,planes)
            #self.gn2 = nn.GroupNorm(32,planes)
            self.gn3 = nn.GroupNorm(32,planes)
            self.bn3 = nn.BatchNorm2d(planes)
        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        # out = self.gn1(x)
        out = self.conv1(x)
        out = self.bn1(out) # out = self.bn1(out)
        out = self.relu(out)
        # out = out + 0.1*out*out

        # out = self.gn2(out)
        out = self.conv2(out)
        out = self.bn2(out) # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)
        if self.modeltype:
            # pdb.set_trace()
            out = out +0.1*out*out
            #pdb.set_trace()
            out = self.gn3(out) # real gn # out = self.bn3(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, modeltype=0, shortcut_norm=None, factor=None, use_relu=None, use_weight=None):
        super(Bottleneck, self).__init__()
        factor = factor
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
        # self.gn1 = nn.GroupNorm(int(32*factor),inplanes, affine=False)
        # self.gn2 = nn.GroupNorm(int(32*factor),planes, affine=False)
        # self.gn3 = nn.GroupNorm(int(32*factor),planes, affine=False)
        if self.modeltype == 2:
            #self.gn1 = nn.GroupNorm(32,planes)
            #self.gn2 = nn.GroupNorm(32,planes)
            #self.gn3 = nn.GroupNorm(32,planes * 4)
            if shortcut_norm == 'gn':
                if use_weight == 1:
                    self.gn4 = nn.GroupNorm(int(32*factor),inplanes, affine=True)
                if use_weight == 0:
                    self.gn4 = nn.GroupNorm(int(32*factor),inplanes, affine=False)
            if shortcut_norm == 'bn':
                if use_weight == 1:
                    self.gn4 = nn.BatchNorm2d(inplanes, affine=True)
                if use_weight == 0:
                    self.gn4 = nn.BatchNorm2d(inplanes, affine=False)
            #self.bn4 = nn.BatchNorm2d(planes * 4)
            #self.avgpool = nn.AvgPool2d(kernel_size = 3, stride=1, padding=1)
            self.gninner = nn.GroupNorm(int(32*factor),planes, affine=True)
            #self.bn4 = nn.BatchNorm2d(planes * 4)
            self.avgpoolinner = nn.AvgPool2d(kernel_size = 1, stride=1, padding=0)
            if stride == 2:
                self.avgpoolinner = nn.AvgPool2d(kernel_size = 3, stride=stride, padding=1)
            pass

        # if self.modeltype == 1:
        #     #self.gn1 = nn.GroupNorm(32,planes)
        #     #self.gn2 = nn.GroupNorm(32,planes)
        #     #self.gn3 = nn.GroupNorm(32,planes * 4)
        #     self.gn4 = nn.GroupNorm(int(32*factor),planes * 4, affine=True)

        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        #residual_inden = x
        #x = self.falserelu(x)
        residual = x


        #out = self.gn1(x)
        #out = 2*x - self.avgpool(x)
        out = self.conv1(x) # out = self.conv1(x) be careful, x or out
        out = self.bn1(out)
        out = self.relu(out)
        #if self.modeltype:
            #out = out +0.01*out*out
            #out = self.gn1(out)

        #out = self.gn2(out)
        residual_inner = out
        out = self.conv2(out)
        out = self.bn2(out)
        if self.modeltype == 2:
           residual_inner = self.gninner(residual_inner)
           residual_inner = self.avgpoolinner(residual_inner)
           out += residual_inner
        out = self.relu(out)
        #if self.modeltype:
            #out = out +0.01*out*out
            #out = self.gn2(out)

        #out = self.gn3(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.modeltype ==2:

            residual = self.gn4(residual) #+ residual_inden
        if self.downsample is not None:
            residual = self.downsample(residual) # x changed to residual
            #if self.stride == 1:
            #	residual = torch.cat((residual, residual, residual, residual), 1)
            #else:
            #	residual = torch.cat((residual, residual), 1)

        if not self.cbam is None:
            out = self.cbam(out)

        #if self.modeltype:
            #residual = self.gn3(residual)
            #pass

        out += residual

        out = self.relu(out) # uncomment
        # if self.modeltype ==1:

        #     out = self.gn4(out)
        return out

class BottleneckSimple(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, modeltype=0, factor=None, use_relu=None):
        super(BottleneckSimple, self).__init__()
        self.use_relu = use_relu
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
        self.gninner = nn.GroupNorm(int(32*factor),planes, affine=True)
        if use_cbam:
            self.cbam = CBAM( planes * 4, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        if self.use_relu == 0: # if relu is not used in the shortcut, it needs to be put in the residual path
            out = self.falserelu(x)
            out = self.conv1(out)
        elif self.use_relu == 1:
            out = self.conv1(x)
        #out = self.conv1(x) # out = self.conv1(x) be careful, x or out
        out = self.bn1(out)
        out = self.relu(out)

        residual_inner = out    

        out = self.conv2(out)
        out = self.bn2(out)
        residual_inner = self.gninner(residual_inner)
        out += residual_inner    
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if not self.cbam is None:
            out = self.cbam(out)

        return out

class ShortcutWeight(nn.Module):
    def __init__(self, num_features, use_weight):
        super(ShortcutWeight, self).__init__()
        if use_weight == 1:
            self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        elif use_weight == 2:
            self.weight = nn.Parameter(torch.ones(1,1,1,1))
      
        #self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))

    def forward(self, x):
        return x * self.weight #+ self.bias

class BNfunction(nn.Module):
    def __init__(self, num_features):
        super(BNfunction, self).__init__()
        #self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        #self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.bn = nn.BatchNorm2d(num_features, affine=True)

    def forward(self, x):
        return self.bn(x) # x * self.weight + self.bias

class ShortcutNormFalse(nn.Module):
    def __init__(self, num_features, shortcut_norm, factor=None):
        super(ShortcutNormFalse, self).__init__()
        #self.weight = nn.Parameter(torch.ones(1,num_features,1,1))
        #self.bias = nn.Parameter(torch.zeros(1,num_features,1,1))
        self.shortcut_norm = shortcut_norm

        if shortcut_norm == 'bn':
            self.shortcut_norm = nn.BatchNorm2d(num_features, affine=False)
        elif shortcut_norm == 'gn':
            self.shortcut_norm = nn.GroupNorm(int(32*factor),num_features, affine=False)

    def forward(self, x):
        if self.shortcut_norm == 'None':
            out = x
        else:
            out = self.shortcut_norm(x)


        return out # self.shortcut_norm(x) # x * self.weight + self.bias

class BottleneckSimpleDense(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, modeltype=0, shortcut_norm=None, factor=None, use_relu=None, use_weight=None , blocks=1):
        super(BottleneckSimpleDense, self).__init__()
        #if blocks == 3:
        self.use_relu = use_relu
        self.use_weight = use_weight
        self.inplanes = inplanes
        self.planes = planes
        self.blocks = blocks
        self.relu = nn.ReLU(inplace=True)
        self.SimpleList = []
        self.ShortcutWeightList = []
        self.ShortcutNormList = []
        self.WeightList = []
        self.BiasList = []
        factor = factor
        self.gn = nn.GroupNorm(int(32*factor),inplanes, affine=False)
        self.bn = nn.BatchNorm2d(inplanes, affine=False)
        for idx in range(blocks-1):
            weight_list = []
            bias_list = []
            ShortcutWeight_list = []

            self.SimpleList.append(BottleneckSimple(self.inplanes, self.planes, stride, downsample, use_cbam=use_cbam, modeltype=modeltype, factor=factor, use_relu=self.use_relu))
            self.ShortcutNormList.append(ShortcutNormFalse(self.inplanes, shortcut_norm, factor))

            if self.use_weight != 0:
                for idy in range(idx+1):
                    self.ShortcutWeightList.append(ShortcutWeight(self.inplanes, self.use_weight))

        self.SimpleList = nn.ModuleList(self.SimpleList)
        self.ShortcutNormList = nn.ModuleList(self.ShortcutNormList)
        if self.use_weight == 1:
            self.ShortcutWeightList = nn.ModuleList(self.ShortcutWeightList)

    def forward(self, x):
        out = x
        NormalizedOutList = []

        index = 0

        for idx in range(self.blocks-1):
            #GnList.append(self.bn(out)) # GnList.append(self.bn(out)) #GnList.append(self.gn(out))
            #GnList.append(out)
            NormalizedOutList.append(self.ShortcutNormList[idx](out))
            simple = self.SimpleList[idx]
            out = simple(out)

            for idy, NormalizedOut in enumerate(NormalizedOutList):
                #out += self.ShortcutWeightList[index](Gn) #Gn*self.WeightList[idx][idy] + self.BiasList[idx][idy].cuda()
                #out += self.BNfunctionList[index](Gn)
                if self.use_weight == 1:
                    out  += self.ShortcutWeightList[index](NormalizedOut)
                elif self.use_weight == 0:
                    out += NormalizedOut
                index += 1
            if self.use_relu == 1:
                out = self.relu(out)
        if self.use_weight:
            assert len(self.ShortcutWeightList) == index

        return out



class ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes, att_type=None, modeltype=0, shortcut_norm=None, factor=None, use_relu=None, use_weight=None, dense=None):
        #self.factor = factor
        self.use_relu = use_relu
        self.inplanes = int(64*factor)
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, int(64*factor), kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, int(64*factor), kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(int(64*factor))
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(int(64*factor)*block.expansion)
            self.bam2 = BAM(int(128*factor)*block.expansion)
            self.bam3 = BAM(int(256*factor)*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.auto = None#nn.Parameter(torch.tensor(0.0))#nn.Parameter(torch.tensor(0.0))


        self.layer1 = self._make_layer(block, int(64*factor),  layers[0], att_type=att_type, modeltype=modeltype, shortcut_norm=shortcut_norm, factor=factor, use_relu=use_relu, use_weight=use_weight, dense=dense)
        self.layer2 = self._make_layer(block, int(128*factor), layers[1], stride=2, att_type=att_type, modeltype=modeltype, shortcut_norm=shortcut_norm, factor=factor, use_relu=use_relu, use_weight=use_weight, dense=dense)
        self.layer3 = self._make_layer(block, int(256*factor), layers[2], stride=2, att_type=att_type, modeltype=modeltype, shortcut_norm=shortcut_norm, factor=factor, use_relu=use_relu, use_weight=use_weight, dense=dense)
        self.layer4 = self._make_layer(block, int(512*factor), layers[3], stride=2, att_type=att_type, modeltype=modeltype, shortcut_norm=shortcut_norm, factor=factor, use_relu=use_relu, use_weight=use_weight, dense=dense)

        self.fc = nn.Linear(int(512*factor) * block.expansion, num_classes)


        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
                #if "gn" in key:
                #    if "SpatialGate" in key:
                #        self.state_dict()[key][...] = 0
                #    else:
                #        self.state_dict()[key][...] = 1
                if 'auto' in key:
                    self.state_dict()[key][...] = 0.0
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None, modeltype = 0, shortcut_norm=None, factor=None, use_relu=None, use_weight=None, dense=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                 nn.BatchNorm2d(planes * block.expansion)
                #nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0),
                #nn.BatchNorm2d(self.inplanes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM', modeltype=modeltype, shortcut_norm=shortcut_norm, factor=factor, use_relu=use_relu, use_weight=use_weight))
        self.inplanes = planes * block.expansion
        if dense == 1:
            layers.append(BottleneckSimpleDense(self.inplanes, planes, use_cbam=att_type=='CBAM', modeltype=modeltype, shortcut_norm=shortcut_norm, factor=factor, use_relu=use_relu, use_weight=use_weight, blocks=blocks))
        elif dense == 0:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM', modeltype=modeltype, shortcut_norm=shortcut_norm, factor=factor, use_relu=use_relu, use_weight=use_weight))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)
        if self.use_relu == 0:  # when it is not used in the shortcut, the final relu should be used
            x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #print(self.auto[0])
        return x

def ResidualNet(network_type, depth, num_classes, att_type, modeltype, shortcut_norm, factor, use_relu, use_weight, dense):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 26, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type, modeltype)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 26:
        model = ResNet(Bottleneck, [2, 2, 2, 2], network_type, num_classes, att_type, modeltype, shortcut_norm, factor, use_relu , use_weight, dense)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type, modeltype, shortcut_norm, factor, use_relu, use_weight, dense)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type, modeltype, shortcut_norm, factor, use_relu, use_weight, dense)

    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResidualNet('ImageNet', 50, 365, None, 2, 'gn', 1.0, 1, 1, 1)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def demo():
    from torchstat import stat
    net = resnet50(num_classes=365)
    y = net(torch.randn(1, 3, 224, 224))
    print(y.size())
    stat(net, (3, 224, 224))

# demo()
