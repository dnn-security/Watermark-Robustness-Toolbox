import torch
import torch.nn as nn
import torch.nn.functional as F


class CifarAutoencoder(nn.Module):
    def __init__(self):
        super(CifarAutoencoder, self).__init__()
        self.enc_1 = nn.Linear(in_features=3072, out_features=768)
        self.enc_2 = nn.Linear(in_features=768, out_features=384)
        self.dec_1 = nn.Linear(in_features=384, out_features=768)
        self.dec_1 = nn.Linear(in_features=768, out_features=3072)

    def forward(self, x):
        x = F.relu(self.enc_1(x.view(-1, 3072)))
        x = torch.sigmoid(self.enc_2(x))
        x = F.relu(self.dec_1(x))
        x = self.dec_2(x).view(-1, 3, 32, 32)
        return x
