# inspired by Frame-level speaker embeddings for text-independent speaker
# recognition and analysis of end-to-end model, Suwon Shon, SLT`18

import torch
import torch.nn as nn
import torch.nn.functional as F

def statistic_pool(x):
    mean = x.mean(2)
    std = x.std(2)
    stat = torch.cat([mean, std], -1)

    return stat

class conv1d_fullbank(nn.Module):
    def __init__(self, config, n_labels):
        """docstring for __init__"""
        super(conv1d_fullbank, self).__init__()
        in_dim = config['input_dim']
        self.conv1d_0 = nn.Conv1d(in_dim, 1000, stride=1, kernel_size=5)
        self.bn_0 = nn.BatchNorm1d(1000)
        self.conv1d_1 = nn.Conv1d(1000, 1000, stride=2, kernel_size=7)
        self.bn_1 = nn.BatchNorm1d(1000)
        self.conv1d_2 = nn.Conv1d(1000, 1000, stride=1, kernel_size=1)
        self.bn_2 = nn.BatchNorm1d(1000)
        self.conv1d_3 = nn.Conv1d(1000, 1500, stride=1, kernel_size=1)
        self.bn_3 = nn.BatchNorm1d(1500)
        self.fc1 = nn.Linear(3000, 1500)
        self.fc2 = nn.Linear(1500, 600)
        self.fc3 = nn.Linear(600, n_labels)

    def embed(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = F.relu(self.bn_0(self.conv1d_0(x)), inplace=True)
        x = F.relu(self.bn_1(self.conv1d_1(x)), inplace=True)
        x = F.relu(self.bn_2(self.conv1d_2(x)), inplace=True)
        x = F.relu(self.bn_3(self.conv1d_3(x)), inplace=True)
        x = statistic_pool(x)
        x = self.fc1(x)
        x = F.relu(self.fc2(x), inplace=True)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.fc3(x)

        return x




