"""
Wide resnet implementation in Pytorch
Courtesy of https://github.com/meliketoy/wide-resnet.pytorch
"""
import operator
from collections import OrderedDict
from itertools import islice

import mlconfig
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def random_permutation(size):
    """ Draw a random permutation of a certain size.
    """
    perm = np.arange(size)
    np.random.shuffle(perm)
    return perm


def inverse_permutation(permutation):
    """ Invert a given permutation.
    """
    inverse_permutation = []
    for i in range(len(permutation)):
        for j, entry in enumerate(permutation):
            if entry == i:
                inverse_permutation.append(j)
                break
    return inverse_permutation

class Sequential(nn.Module):

    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

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

    def forward(self, input):
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


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        # self.shortcut = nn.Sequential()
        self.shortcut = Sequential()
        self.shortcut_layer = [None]
        if stride != 1 or in_planes != planes:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            # )
            self.shortcut_layer = [nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)]
            self.shortcut = Sequential(
                self.shortcut_layer[0],
            )

        self.shuffle_dict = {
            "bn1": np.arange(in_planes),
            "conv1": np.arange(planes),
            "bn2": np.arange(planes),
            "conv2": np.arange(planes),
            "shortcut": np.arange(planes)
        }

    def shuffle_weights(self):
        for name, layer in {
            "conv1": self.conv1,
            "conv2": self.conv2,
            "shortcut": self.shortcut_layer[0]
        }.items():
            if layer is not None:
                permutation = random_permutation(len(self.shuffle_dict[name]))
                self.shuffle_dict[name] = inverse_permutation(permutation)
                layer.weight.data = layer.weight.data[permutation]

    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]

        bn1 = self.bn1(x)
        conv1 = self.conv1(F.relu(bn1))[:, self.shuffle_dict["conv1"]]

        bn2 = self.bn2(self.dropout(conv1))
        conv2 = self.conv2(F.relu(bn2))[:, self.shuffle_dict["conv2"]]

        skip = conv2 + self.shortcut(x)[-1][:, self.shuffle_dict["shortcut"]]
        return [conv1, conv2, skip]


class WideResNet(nn.Module):
    def __init__(self, n, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

        self.shuffle_dict = {
            "conv1": np.arange(len(self.conv1.weight.data))
        }

    def shuffle_weights(self):
        """ Shuffles the weight of this neural network without changing its functionality.
        """
        # Shuffle conv1.
        permutation = random_permutation(len(self.conv1.weight.data))
        self.shuffle_dict["conv1"] = inverse_permutation(permutation)
        self.conv1.weight.data = self.conv1.weight.data[permutation]

        for layer in [self.layer1, self.layer2, self.layer3]:
            for module in layer.modules():
                if type(module) is wide_basic:
                    module.shuffle_weights()

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return Sequential(*layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.conv1(x)  # 64x16x32x32
        conv1 = conv1[:, self.shuffle_dict["conv1"]]    # Shuffle activations back to normal.

        conv2 = self.layer1(conv1)  # list(64x96x32x32)*3
        conv3 = self.layer2(conv2)  # list(64x192x16x16)*3
        conv4 = self.layer3(conv3)  # list(64x384x16x16)*3
        out = F.relu(self.bn1(conv4[-1]))
        #out = F.relu(self.bn1(conv4))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        #return [conv1, conv2, conv3, conv4, out]
        return [conv1] + conv2 + conv3 + conv4 + [out]


class WideResNet_ArgMax(nn.Module):
    def __init__(self, n, widen_factor, dropout_rate, num_classes):
        super(WideResNet_ArgMax, self).__init__()
        self.in_planes = 16

        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return Sequential(*layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.layer1(conv1)
        conv3 = self.layer2(conv2)
        conv4 = self.layer3(conv3)
        out = F.relu(self.bn1(conv4[-1]))
        #out = F.relu(self.bn1(conv4))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


@mlconfig.register
def cifar_wide_resnet(**kwargs):
    model = WideResNet_ArgMax(2, 6, 0, 10)
    model.apply(conv_init)
    return model


@mlconfig.register
def cifar_wide_resnet_features(**kwargs):
    model = WideResNet(2, 6, 0, 10)
    model.apply(conv_init)
    return model

