import torch
import torch.nn as nn
import torch.nn.functional as F


from .model import SerializableModule, num_flat_features
from .model import AngleLinear


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
    if pool_size > 1:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_size)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

class SimpleCNN(SerializableModule):
    def __init__(self, config, n_labels):
        super().__init__()
        input_frames = config["splice_frames"]
        hid_dim = 64
        in_dim = config["input_dim"]
        self.feat_size = hid_dim
        self.convb_1 = conv_block(1, hid_dim)
        self.convb_2 = conv_block(hid_dim, hid_dim)
        self.convb_3 = conv_block(hid_dim, hid_dim)
        if input_frames < 21:
            self.convb_4 = conv_block(hid_dim, hid_dim, 1)
        else:
            self.convb_4 = conv_block(hid_dim, hid_dim)
        with torch.no_grad():
            test_in = torch.zeros((1, 1, input_frames, in_dim))
            test_out = self.embed(test_in)
            self.feat_dim = test_out.size(1)
            self.output = nn.Linear(self.feat_dim, n_labels)

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
        in_dim = config["input_dim"]
        self.convb_1 = _conv_bn_relu(1, 32, (7, 3), (2, 1))
        self.convb_2 = _conv_bn_relu(32, 32, (7, 3), (2, 1))
        self.convb_3= _conv_bn_relu(32, 64, (7, 3), (2, 1))
        with torch.no_grad():
            x = torch.zeros((1, 1, config["input_frames"], in_dim))
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

class Conv4_2dim(SerializableModule):
    def __init__(self, config, n_labels):
        super().__init__()
        loss_type = config["loss"]
        input_frames = config["splice_frames"][-1]
        hid_dim = 64
        in_dim = config["input_dim"]
        self.convb_1 = conv_block(1, hid_dim)
        self.convb_2 = conv_block(hid_dim, hid_dim, 2)
        self.convb_3 = conv_block(hid_dim, hid_dim)
        self.convb_4 = conv_block(hid_dim, hid_dim, 2)
        # self.convb_2 = conv_block(hid_dim, hid_dim*2, 2)
        # self.convb_3 = conv_block(hid_dim*2, hid_dim*3)
        # self.convb_4 = conv_block(hid_dim*3, hid_dim*3, 2)

        with torch.no_grad():
            test_in = torch.zeros(
                    (1, 1, input_frames, in_dim))
            x = self.convb_1(test_in)
            x = self.convb_2(x)
            x = self.convb_3(x)
            x = self.convb_4(x)
            test_out = x.view(-1, num_flat_features(x))
            self.feat_dim = test_out.size(1)
            self.fc = nn.Linear(self.feat_dim, 2)
            if loss_type == "angular":
                self.output = AngleLinear(2, n_labels, m=4)
            else:
                self.output = nn.Linear(2, n_labels)

    def embed(self, x):
        if x.dim() == 3:
            x = torch.unsqueeze(x, 1)
        x = self.convb_1(x)
        x = self.convb_2(x)
        x = self.convb_3(x)
        x = self.convb_4(x)
        x = x.view(-1, num_flat_features(x))
        x = self.fc(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.output(x)
        return x

class sphere20a(nn.Module):
    def __init__(self, config,  n_labels):
        super(sphere20a, self).__init__()
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(1,32,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(32)
        self.conv1_2 = nn.Conv2d(32,32,3,1,1)
        self.relu1_2 = nn.PReLU(32)
        self.conv1_3 = nn.Conv2d(32,32,3,1,1)
        self.relu1_3 = nn.PReLU(32)

        self.conv2_1 = nn.Conv2d(32,64,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(64)
        self.conv2_2 = nn.Conv2d(64,64,3,1,1)
        self.relu2_2 = nn.PReLU(64)
        self.conv2_3 = nn.Conv2d(64,64,3,1,1)
        self.relu2_3 = nn.PReLU(64)

        self.conv2_4 = nn.Conv2d(64,64,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(64)
        self.conv2_5 = nn.Conv2d(64,64,3,1,1)
        self.relu2_5 = nn.PReLU(64)


        self.conv3_1 = nn.Conv2d(64, 128,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(128)
        self.conv3_2 = nn.Conv2d(128,128,3,1,1)
        self.relu3_2 = nn.PReLU(128)
        self.conv3_3 = nn.Conv2d(128,128,3,1,1)
        self.relu3_3 = nn.PReLU(128)

        self.conv3_4 = nn.Conv2d(128,128,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(128)
        self.conv3_5 = nn.Conv2d(128,128,3,1,1)
        self.relu3_5 = nn.PReLU(128)

        self.conv3_6 = nn.Conv2d(128,128,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(128)
        self.conv3_7 = nn.Conv2d(128,128,3,1,1)
        self.relu3_7 = nn.PReLU(128)

        self.conv3_8 = nn.Conv2d(128,128,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(128)
        self.conv3_9 = nn.Conv2d(128,128,3,1,1)
        self.relu3_9 = nn.PReLU(128)

        self.conv4_1 = nn.Conv2d(128,256,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(256)
        self.conv4_2 = nn.Conv2d(256,256,3,1,1)
        self.relu4_2 = nn.PReLU(256)
        self.conv4_3 = nn.Conv2d(256,256,3,1,1)
        self.relu4_3 = nn.PReLU(256)

        # self.fc5 = nn.Linear(256*16*4,256)
        self.fc6 = AngleLinear(256, n_labels)


    def embed(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = F.avg_pool2d(x, x.shape[-2:])
        x = x.view(x.size(0),-1)
        # x = self.fc5(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.fc6(x)

        return x

