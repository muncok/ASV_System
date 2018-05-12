import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .model import SerializableModule, num_flat_features

class voxNet(SerializableModule):
    def __init__(self, nb_class):
        super(voxNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 96, 7, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 256, 3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)
        self.fc6 = nn.Conv2d(256, 4096, (9, 1))
        self.fc6_bn = nn.BatchNorm2d(4096)
        self.fc7 = nn.Linear(4096, 1024)
        self.fc7_bn = nn.BatchNorm2d(1024)
        self.fc8 = nn.Linear(1024, nb_class)
        self.feat_size = 1024

    def embed(self, x):
        # x: (512, 300) = (freq, windows)
        windows_width = x.size(-1)
        if x.dim() == 2:
            x = x.view(1,1,x.size(0),x.size(1))
        if x.dim() == 3:
            x = torch.unsqueeze(x, 1)
        net = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x))), 3, 2)
        net = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(net))), 3, 2)
        net = F.relu(self.conv3_bn(self.conv3(net)))
        net = F.relu(self.conv4_bn(self.conv4(net)))
        net = F.max_pool2d(F.relu(self.conv5_bn(self.conv5(net))), [5, 3], stride=[3, 2])
        support = 3 * (windows_width // 100) - 1
        net = (F.avg_pool2d(F.relu(self.fc6_bn(self.fc6(net))), [1, support]))
        net = net.view(-1, num_flat_features(net))
        fc7_out = F.relu(self.fc7_bn(self.fc7(net)))
        return fc7_out

    def forward(self, x):
        x = self.embed(x)
        fc8_out = self.fc8(x)
        return fc8_out


def conv_block(in_channels, out_channels, pool_size=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(pool_size)
    )

class SimpleCNN(SerializableModule):
    def __init__(self, config, n_labels):
        super().__init__()
        input_frames = config["splice_frames"]
        hid_dim = 64
        self.feat_size = 64
        self.convb_1 = conv_block(1, hid_dim)
        self.convb_2 = conv_block(hid_dim, hid_dim)
        self.convb_3 = conv_block(hid_dim, hid_dim)
        if input_frames < 21:
            self.convb_4 = conv_block(hid_dim, hid_dim, 1)
        else:
            self.convb_4 = conv_block(hid_dim, hid_dim)

        with torch.no_grad():
            test_in = torch.zeros((1, 1, input_frames, 40))
            test_out = self.embed(test_in)
            self.output = nn.Linear(test_out.size(1), n_labels)
        self.embed_mode = False

    def embed(self, x):
        if x.dim() == 3:
            x = torch.unsqueeze(x, 1)
        x = self.convb_1(x)
        x = self.convb_2(x)
        x = self.convb_3(x)
        x = self.convb_4(x)
        x = x.view(-1, num_flat_features(x))
        return x

    def forward(self, x):
        x = self.embed(x)
        if not self.embed_mode:
            x = self.output(x)
        return x

def _conv_bn_relu(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class LongCNN(SerializableModule):
    def __init__(self, config, n_labels):
        super().__init__()
        self.convb_1 = _conv_bn_relu(1, 32, (7, 3), (2, 1))
        self.convb_2 = _conv_bn_relu(32, 32, (7, 3), (2, 1))
        self.convb_3= _conv_bn_relu(32, 64, (7, 3), (2, 1))
        with torch.no_grad():
            x = torch.zeros((1, 1, config["splice_frames"], 40))
            x = self.embed(x)
        self.output = nn.Linear(x.size(1), n_labels)

    def embed(self, x):
        if x.dim() == 3:
            x = torch.unsqueeze(x, 1)
        x = self.convb_1(x)
        x = self.convb_2(x)
        x = self.convb_3(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(-1, num_flat_features(x))
        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.output(x)
        return x







