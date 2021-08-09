import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistAutoencoder(nn.Module):
    def __init__(self):
        super(MnistAutoencoder, self).__init__()
        # self.enc_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # self.enc_2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        # self.enc_3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        #
        # self.dec_1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        # self.dec_2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # self.dec_3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

        self.enc_1 = nn.Linear(in_features=784, out_features=196)
        self.dec_1 = nn.Linear(in_features=196, out_features=784)

    def forward(self, x):
        x = torch.sigmoid(self.enc_1(x.view(-1, 784)))
        x = self.dec_1(x).view(-1, 1, 28, 28)

        # x = F.max_pool2d(F.relu(self.enc_1(x)), 2)
        # x = F.max_pool2d(F.relu(self.enc_2(x)), 2)
        # x = torch.sigmoid(self.enc_3(x))
        # x = F.relu(self.dec_1(x))
        # x = F.relu(self.dec_2(F.upsample(x, size=14)))
        # x = self.dec_3(F.upsample(x, size=28))

        return x
