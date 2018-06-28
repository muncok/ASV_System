import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock

from .commons import SerializableModule, num_flat_features, conv_block
from .Angular_loss import AngleLinear

class ResNet34(ResNet):
    def __init__(self, config, layers, n_labels=1000):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 16, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, layers[3], stride=2)
        self.fc = nn.Linear(128 * BasicBlock.expansion, 128)
        loss_type = config["loss"]
        if loss_type == "angle":
            self.output = AngleLinear(128, n_labels)
        else:
            self.output = nn.Linear(128, n_labels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print("saved to {}".format(filename))

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        print("loaded from {}".format(filename))

    def load_partial(self, filename):
        to_state = self.state_dict()
        from_state = torch.load(filename)
        valid_state = {k:v for k,v in from_state.items() if k in to_state.keys()}
        valid_state.pop('output.weight', None)
        valid_state.pop('output.bias', None)
        to_state.update(valid_state)
        self.load_state_dict(to_state)
        assert(len(valid_state) > 0)
        print("loaded from {}".format(filename))

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,x.shape[-2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.output(x)

        return x

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



class SimpleCNN(SerializableModule):
    def __init__(self, config, n_labels):
        super().__init__()
        input_frames = config["splice_frames"]
        hid_dim = 64
        in_dim = config["input_dim"]
        self.feat_size = 64
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
            x = torch.zeros((1, 1, config["splice_frames"], in_dim))
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

class Conv4Angle(SerializableModule):
    def __init__(self, config, n_labels):
        super().__init__()
        loss_type = config["loss"]
        input_frames = config["splice_frames"]
        hid_dim = 64
        in_dim = config["input_dim"]
        self.convb_1 = conv_block(1, hid_dim)
        # self.convb_2 = conv_block(hid_dim, hid_dim, 2)
        # self.convb_3 = conv_block(hid_dim, hid_dim)
        # self.convb_4 = conv_block(hid_dim, hid_dim, 2)
        self.convb_2 = conv_block(hid_dim, hid_dim*2, 2)
        self.convb_3 = conv_block(hid_dim*2, hid_dim*3)
        self.convb_4 = conv_block(hid_dim*3, hid_dim*3, 2)

        with torch.no_grad():
            test_in = torch.zeros(
                    (1, 1, input_frames, in_dim))
            x = self.convb_1(test_in)
            x = self.convb_2(x)
            x = self.convb_3(x)
            x = self.convb_4(x)
            test_out = x.view(-1, num_flat_features(x))
            self.feat_dim = test_out.size(1)
            self.fc = nn.Linear(self.feat_dim,512)
            if loss_type == "angle":
                self.output = AngleLinear(512, n_labels, m=4)
            else:
                self.output = nn.Linear(512, n_labels)

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
