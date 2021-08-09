import mlconfig
import torchvision

import os

from wrt.training.models.torch.classifier.tf_resnet import resnet101


@mlconfig.register
def resnet34(pretrained=False, **kwargs):
    return torchvision.models.resnet34(pretrained=pretrained)


@mlconfig.register
def resnet_pretrained_open_images(freeze_first_n_layers=312, **kwargs):
    pretrained_path = os.path.abspath(os.path.join('outputs', 'imagenet', 'wm', 'pretrained', 'oid_resnet101.pt'))
    model = resnet101(num_classes=5000)
    model.load_state_dict(torch.load(pretrained_path))
    model = torch.nn.DataParallel(model)

    for params in list(model.parameters())[:freeze_first_n_layers]:
        params.requires_grad = False

    return model

"""
Code from https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html
"""

from collections import OrderedDict
from itertools import islice
import operator

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Sequential(nn.Module):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

        self.return_hidden_activations = False

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module) -> None:
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self) -> int:
        return len(self._modules)

    def __dir__(self):
        keys = super(Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self):
        return iter(self._modules.values())

    def __forward_single(self, input):
        for module in self:
            input = module(input)
        return input

    def __forward_multi(self, input):
        if isinstance(input, list):
            input = input[-1]

        outputs = []
        non_empty = False
        for module in self:
            non_empty = True
            input = module(input)
            if isinstance(input, list):
                outputs.extend(input)
            else:
                outputs.append(input)
        if not non_empty:
            return [input]
        else:
            return outputs

    def forward(self, input):
        if self.return_hidden_activations:
            return self.__forward_multi(input)
        else:
            return self.__forward_single(input)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self._return_hidden_activations = False

    @property
    def return_hidden_activations(self):
        return self._return_hidden_activations

    @return_hidden_activations.setter
    def return_hidden_activations(self, value):
        self._return_hidden_activations = value
        if self.downsample is not None:
            self.downsample.return_hidden_activations = value

    def __forward_single(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def __forward_multi(self, x):
        if isinstance(x, list):
            x = x[-1]

        conv1 = self.conv1(x)
        conv2 = self.conv2(self.relu(self.bn1(conv1)))
        conv3 = self.conv3(self.relu(self.bn2(conv2)))

        if self.downsample is not None:
            x = self.downsample(x)[-1]

        out = self.relu(self.bn3(conv3) + x)

        return [conv1, conv2, conv3, out]

    def forward(self, x):
        if self._return_hidden_activations:
            return self.__forward_multi(x)
        else:
            return self.__forward_single(x)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._return_hidden_activations = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    @property
    def return_hidden_activations(self):
        return self._return_hidden_activations

    @return_hidden_activations.setter
    def return_hidden_activations(self, value):
        self._return_hidden_activations = value
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            layer.return_hidden_activations = value
            for block in layer:
                block.return_hidden_activations = value

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return Sequential(*layers)

    def __forward_single(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def __forward_multi(self, x):
        conv0 = self.conv1(x)
        conv1 = self.layer1(self.maxpool(self.relu(self.bn1(conv0))))
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)

        out = self.fc(torch.flatten(self.avgpool(conv4[-1]), 1))
        return [conv0] + conv1 + conv2 + conv3 + conv4 + [out]

    def forward(self, x):
        if self._return_hidden_activations:
            return self.__forward_multi(x)
        else:
            return self.__forward_single(x)


def ImageNetWRTModel():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model


@mlconfig.register
def imagenet_resnet(dropout=0, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3])
