import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from wrt.classifiers import PyTorchClassifier


class MnistAutoencoder(nn.Module):

    def __init__(self, layer_size):
        super().__init__()
        self.fc_1 = nn.Linear(in_features=784, out_features=layer_size)
        self.fc_2 = nn.Linear(in_features=layer_size, out_features=784)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc_1(x))
        x = torch.sigmoid(self.fc_2(x)).view(-1, 1, 28, 28)
        return x


class CifarAutoencoder(nn.Module):

    def __init__(self, num_filters):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(num_features=num_filters)
        self.conv_2 = nn.Conv2d(in_channels=num_filters, out_channels=3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.bn(self.conv_1(x)))
        x = torch.sigmoid(self.conv_2(F.upsample(x, 32)))
        return x


def mnist_autoencoder(layer_size):
    model = MnistAutoencoder(layer_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.MSELoss()
    autoencoder = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        clip_values=(0, 1)
    )
    return autoencoder


def cifar_autoencoder(num_filters):
    model = CifarAutoencoder(num_filters)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss = nn.MSELoss()
    autoencoder = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0, 1)
    )
    return autoencoder


# adapted from https://github.com/foamliu/Autoencoder

class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True))
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class SegNet(nn.Module):
    def __init__(self, planes=64):
        super(SegNet, self).__init__()

        self.down1 = segnetDown2(3, planes)
        self.down2 = segnetDown2(planes, planes * 2)
        self.down3 = segnetDown3(planes * 2, planes * 4)
        self.down4 = segnetDown3(planes * 4, planes * 8)
        self.down5 = segnetDown3(planes * 8, planes * 8)

        self.up5 = segnetUp3(planes * 8, planes * 8)
        self.up4 = segnetUp3(planes * 8, planes * 4)
        self.up3 = segnetUp3(planes * 4, planes * 2)
        self.up2 = segnetUp2(planes * 2, planes)
        self.up1 = segnetUp2(planes, 3)

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        return up1


def imagenet_autoencoder(num_filters):
    model = SegNet(num_filters)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()
    autoencoder = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
        clip_values=(0, 1)
    )
    return autoencoder
