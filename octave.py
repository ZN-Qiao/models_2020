import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['oct_resnet50']

class OctConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 alpha=0.5,
                 dilation=1,
                 groups=False,
                 bias=False):

        """
        Octave convolution from the 2019 article
        Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution
        Extend the 2D convolution with the octave reduction.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            alpha (float or tuple, optional): Reduction for the (input, output) octave part of the convolution.
                Default: 0.5
            groups (bool, optional): Decides if the convolution must be group-wise, with groups=in_channels.
                Default: False
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``False``
        """

        super(OctConv2d, self).__init__()

        assert isinstance(in_channels, int) and in_channels > 0
        assert isinstance(out_channels, int) and out_channels > 0
        assert isinstance(kernel_size, int) and kernel_size > 0
        assert stride in {1, 2}, "Only strides of 1 and 2 are currently supported"

        if isinstance(alpha, tuple):
            assert len(alpha) == 2
            assert all([0 <= a <= 1 for a in alpha]), "Alphas must be in interval [0, 1]"
            self.alpha_in, self.alpha_out = alpha
        else:
            assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
            self.alpha_in = alpha
            self.alpha_out = alpha

        # in_channels
        in_ch_hf = int((1 - self.alpha_in) * in_channels)
        self.in_channels = {
            'high': in_ch_hf,
            'low': in_channels - in_ch_hf
        }

        # out_channels
        out_ch_hf = int((1 - self.alpha_out) * out_channels)
        self.out_channels = {
            'high': out_ch_hf,
            'low': out_channels - out_ch_hf
        }

        # groups
        self.groups = {
            'high': 1,
            'low': 1
        }

        if type(groups) == bool and groups:
            if self.alpha_out > 0 and self.in_channels['high'] <= self.out_channels['high']:
                self.groups['high'] = in_ch_hf

            if self.alpha_in > 0 and self.in_channels['low'] <= self.out_channels['low']:
                self.groups['low'] = in_channels - in_ch_hf

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.conv_h2h = nn.Conv2d(in_channels=self.in_channels['high'],
                                  out_channels=self.out_channels['high'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=self.groups['high'],
                                  bias=bias) \
            if not (self.alpha_in == 1 or self.alpha_out == 1) else None

        self.conv_h2l = nn.Conv2d(in_channels=self.in_channels['high'],
                                  out_channels=self.out_channels['low'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=self.groups['high'],
                                  bias=bias) \
            if not (self.alpha_in == 1 or self.alpha_out == 0) else None

        self.conv_l2h = nn.Conv2d(in_channels=self.in_channels['low'],
                                  out_channels=self.out_channels['high'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=self.groups['low'],
                                  bias=bias) \
            if not (self.alpha_in == 0 or self.alpha_out == 1) else None

        self.conv_l2l = nn.Conv2d(in_channels=self.in_channels['low'],
                                  out_channels=self.out_channels['low'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  dilation=dilation,
                                  groups=self.groups['low'],
                                  bias=bias) \
            if not (self.alpha_in == 0 or self.alpha_out == 0) else None

    def forward(self, x):
        x_h, x_l = x if isinstance(x, tuple) else (x, None)

        self._check_inputs(x_h, x_l)

        x_l2l, x_l2h = None, None

        # High -> High
        x_h = self.pool(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h) if self.out_channels['high'] > 0 else None

        # High -> Low
        x_h2l = self.pool(x_h) if self.out_channels['low'] > 0 else x_h
        x_h2l = self.conv_h2l(x_h2l) if self.out_channels['low'] > 0 else None

        if x_l is not None:
            # Low -> Low
            x_l2l = self.pool(x_l) if (self.out_channels['low'] > 0 and self.stride == 2) else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.out_channels['low'] > 0 else None

            # Low -> High
            x_l2h = self.conv_l2h(x_l) \
                if (self.out_channels['high'] > 0 and self.in_channels['low'] > 0) \
                else None
            x_l2h = F.interpolate(x_l2h, size=x_h2h.shape[-2:]) \
                if (self.out_channels['high'] > 0 and self.stride == 1) else x_l2h

        x_h = x_h2h + x_l2h if x_l2h is not None else x_h2h
        x_l = x_l2l + x_h2l if x_l2l is not None else x_h2l

        output = (x_h, x_l)

        return output[0] if output[1] is None else output

    def _check_inputs(self, x_h, x_l):
        assert x_h.dim() == 4

        if x_l is not None:
            assert x_l.dim() == 4

        if self.in_channels['high'] > 0:
            assert x_h.shape[1] == self.in_channels['high']

        if self.in_channels['low'] > 0:
            assert x_l.shape[1] == self.in_channels['low']

    def __repr__(self):
        s = """{}(in_channels=(low: {}, high: {}), out_channels=(low: {}, high: {}),
          kernel_size=({kernel}, {kernel}), stride=({stride}, {stride}),
          padding={}, alphas=({}, {}), dilation={dilation}, groups=(low: {groupsl}, high: {groupsh}),
          bias={})""".format(
            self._get_name(), self.in_channels['low'], self.in_channels['high'],
            self.out_channels['low'], self.out_channels['high'],
            self.padding, self.alpha_in, self.alpha_out, self.bias,
            kernel=self.kernel_size, stride=self.stride, dilation=self.dilation,
            groupsl=self.groups['low'], groupsh=self.groups['high'])

        return s



class OctConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=0,
                 bias=False, norm_layer=None):

        super(OctConvBn, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = OctConv2d(in_channels, out_channels, kernel_size=kernel_size,
                              alpha=alpha, stride=stride, padding=padding, bias=bias)

        alpha_out = self.conv.alpha_out

        self.bn_h = None if alpha_out == 1 else norm_layer(self.conv.out_channels['high'])
        self.bn_l = None if alpha_out == 0 else norm_layer(self.conv.out_channels['low'])

    def forward(self, x):
        out = self.conv(x)

        x_h, x_l = out if isinstance(out, tuple) else (out, None)

        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None

        return x_h, x_l


class OctConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=0,
                 bias=False, norm_layer=None, activation_layer=None):

        super(OctConvBnAct, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation_layer is None:
            activation_layer = nn.ReLU(inplace=True)

        self.conv = OctConv2d(in_channels, out_channels, kernel_size=kernel_size,
                              alpha=alpha, stride=stride, padding=padding, bias=bias)

        alpha_out = self.conv.alpha_out

        self.bn_h = None if alpha_out == 1 else norm_layer(self.conv.out_channels['high'])
        self.bn_l = None if alpha_out == 0 else norm_layer(self.conv.out_channels['low'])

        self.act = activation_layer

    def forward(self, x):
        out = self.conv(x)

        x_h, x_l = out if isinstance(out, tuple) else (out, None)

        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None

        return x_h, x_l


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, alpha=0.5, norm_layer=None,
                 first_block=False, last_block=False):

        super(Bottleneck, self).__init__()

        assert not (first_block and last_block), "mutually exclusive options"

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = OctConvBnAct(inplanes, width, kernel_size=1, norm_layer=norm_layer,
                                  alpha=alpha if not first_block else (0., alpha))
        self.conv2 = OctConvBnAct(width, width, kernel_size=3, stride=stride, padding=1,
                                  norm_layer=norm_layer, alpha=alpha)
        self.conv3 = OctConvBn(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer,
                               alpha=alpha if not last_block else (alpha, 0.))

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        # The first and last two convs don't have identity_l

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        out = self.conv3((x_h, x_l))

        x_h, x_l = out if isinstance(out, tuple) else (out, None)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity_h, identity_l = identity if isinstance(identity, tuple) else (identity, None)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l


class OctResNet(nn.Module):

    def __init__(self, block, layers, num_classes=365, groups=1, width_per_group=64, norm_layer=None, alpha=0.5):
        super(OctResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.alpha = alpha
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, first_layer=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, last_layer=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, first_layer=False, last_layer=False):

        assert not (first_layer and last_layer), "mutually exclusive options"

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if last_layer:
                downsample = nn.Sequential(
                    OctConvBn(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                              alpha=(self.alpha, 0.))
                )
            else:
                downsample = nn.Sequential(
                    OctConvBn(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                              alpha=self.alpha if not first_layer else (0., self.alpha))
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,
                            groups=self.groups, base_width=self.base_width,
                            alpha=self.alpha, norm_layer=norm_layer,
                            first_block=first_layer, last_block=last_layer))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha=self.alpha if not last_layer else 0.,
                                last_block=last_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_h, x_l = self.layer1(x)
        x_h, x_l = self.layer2((x_h, x_l))
        x_h, x_l = self.layer3((x_h, x_l))
        x_h, x_l = self.layer4((x_h, x_l))

        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _oct_resnet(inplanes, planes, **kwargs):
    model = OctResNet(inplanes, planes, **kwargs)
    return model


def oct_resnet50(**kwargs):
    """Constructs a OctResNet-50 model."""
    return _oct_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def oct_resnet101(**kwargs):
    """Constructs a OctResNet-101 model."""
    return _oct_resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def oct_resnet152(**kwargs):
    """Constructs a OctResNet-152 model."""
    return _oct_resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


def demo():
    from torchstat import stat

    net = oct_resnet50()
    y = net(torch.randn(1, 3, 224, 224))
    print(y.size())
    stat(net, (3, 224, 224))

# demo()
