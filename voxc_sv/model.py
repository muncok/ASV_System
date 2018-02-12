import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #torch.nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.kaiming_normal(m.weight, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

class SpeechResModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_maps = config["n_feature_maps"]
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        if "res_pool" in config:
            self.pool = nn.AvgPool2d(config["res_pool"])

        self.n_layers = n_layers = config["n_layers"]
        dilation = config["use_dilation"]
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.output = nn.Linear(n_maps, n_labels)

    def forward(self, x):
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)

class voxNet(nn.Module):
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
        self.fc6 = nn.Conv2d(256, 4096, [9, 1])
        self.fc6_bn = nn.BatchNorm2d(4096)
        self.fc7 = nn.Linear(4096, 1024)
        self.fc7_bn = nn.BatchNorm2d(1024)
        self.fc8 = nn.Linear(1024, nb_class)

    def forward(self, x, output_l = 'fc8'):
        # x: (512, 300) = (freq, windows)
        windows_width = x.size()[-1]
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
        fc8_out = self.fc8(fc7_out)

        if output_l == 'fc7':
            return fc7_out
        else:
            return fc8_out


class mVoxNet(nn.Module):
    def __init__(self, nb_class):
        super(mVoxNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.fc6 = nn.Conv2d(128, 512, [4, 1])
        self.fc6_bn = nn.BatchNorm2d(512)
        self.fc7 = nn.Linear(512, 256)
        self.fc7_bn = nn.BatchNorm2d(256)
        self.fc8 = nn.Linear(256, nb_class)

    def forward(self, x, output_l = 'fc8'):
        # x: (512, 300) = (freq, windows)
        windows_width = x.size()[-1]
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
        fc8_out = self.fc8(fc7_out)

        if output_l == 'fc7':
            return fc7_out
        else:
            return fc8_out

class siameseNet(nn.Module):
    def __init__(self, feature_size=256):
        super(siameseNet, self).__init__()
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
        self.fc6 = nn.Conv2d(256, 4096, [9, 1])
        self.fc6_bn = nn.BatchNorm2d(4096)
        self.fc7 = nn.Linear(4096, 1024)
        self.fc7_bn = nn.BatchNorm2d(1024)
        self.fc8 = nn.Linear(1024, feature_size)


    def forward_once(self, x, output_l='fc8'):
        # x: (512, 300) = (freq, windows)
        windows_width = x.size()[-1]
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
        fc7_out = fc7_out / fc7_out.norm(2)
        fc8_out = self.fc8(fc7_out)

        if output_l == 'fc7':
            return fc7_out
        else:
            return fc8_out

    def forward(self, x0, x1, output_l='fc8'):
        output0  = self.forward_once(x0, output_l)
        output1  = self.forward_once(x1, output_l)
        return output0, output1


class speechNet(nn.Module):
    def __init__(self, nb_uttrs, nb_class):
        super(speechNet, self).__init__()
        #### conv-1 ####
        self.conv11 = nn.Conv3d(1, 16, [3, 1, 5], stride=1)
        self.conv11_bn = nn.BatchNorm2d(16)
        self.prelu11 = nn.PReLU()
        self.conv12 = nn.Conv3d(16, 16, [3, 9, 1], stride=(1, 2, 1))
        self.conv12_bn = nn.BatchNorm2d(16)
        self.prelu12 = nn.PReLU()
        #### conv-2 ####
        self.conv21 = nn.Conv3d(16, 32, [3, 1, 4], stride=1, padding=0)
        self.conv21_bn = nn.BatchNorm2d(32)
        self.prelu21 = nn.PReLU()
        self.conv22 = nn.Conv3d(32, 32, [3, 8, 1], stride=(1, 2, 1), padding=0)
        self.conv22_bn = nn.BatchNorm2d(32)
        self.prelu22 = nn.PReLU()
        #### conv-3 ####
        self.conv31 = nn.Conv3d(32, 64, [3, 1, 3], stride=1, padding=0)
        self.conv31_bn = nn.BatchNorm2d(64)
        self.prelu31 = nn.PReLU()
        self.conv32 = nn.Conv3d(64, 64, [3, 7, 1], stride=1, padding=0)
        self.conv32_bn = nn.BatchNorm2d(64)
        self.prelu32 = nn.PReLU()
        #### conv-4 ####
        self.conv41 = nn.Conv3d(64, 128, [3, 1, 3], stride=1, padding=0)
        self.conv41_bn = nn.BatchNorm2d(128)
        self.prelu41 = nn.PReLU()
        self.conv42 = nn.Conv3d(128, 128, [3, 7, 1], stride=1, padding=0)
        self.conv42_bn = nn.BatchNorm2d(128)
        self.prelu42 = nn.PReLU()
        #### conv-5 ####
        self.fc5 = nn.Conv3d(128, 128, [4, 3, 3], stride=1, padding=0)
        self.prelu5 = nn.PReLU()

        self.last_fc = nn.Linear(128, nb_class)



    def forward(self, x):
        # x: (batch_size, C, nb_uttrs, H, W) = (batch_size, channels, uttrs, freq, windows)
        if x.dim() == 4:
            x = torch.unsqueeze(x, 1)

        net = self.prelu11(self.conv11_bn(self.conv11(x)))
        net = F.max_pool3d(self.prelu12((self.conv12_bn(self.conv12(net)))), [1, 1, 2], [1, 1, 2])

        net = self.prelu21(self.conv21_bn(self.conv21(net)))
        net = F.max_pool3d(self.prelu22((self.conv22_bn(self.conv22(net)))), [1, 1, 2], [1, 1, 2])

        net = self.prelu31(self.conv31_bn(self.conv31(net)))
        net = self.prelu32(self.conv32_bn(self.conv32(net)))

        net = self.prelu41(self.conv41_bn(self.conv41(net)))
        net = self.prelu42(self.conv42_bn(self.conv42(net)))

        net = self.prelu5((self.fc5(net)))
        net = net.view(-1, num_flat_features(net))
        net = self.last_fc(net)

        return net




class speechNetTest(nn.Module):
    def __init__(self, nb_uttrs, nb_class):
        super(speechNetTest, self).__init__()
        #### conv-1 ####
        # filter = [uttrs, frames, features]
        self.conv11 = nn.Conv3d(1, 16, [3, 1, 5], stride=1)
        self.conv11_bn = nn.BatchNorm2d(16)
        self.prelu11 = nn.PReLU()
        self.conv12 = nn.Conv3d(16, 16, [3, 9, 1], stride=(1, 2, 1))
        self.conv12_bn = nn.BatchNorm2d(16)
        self.prelu12 = nn.PReLU()
        #### conv-2 ####
        self.conv21 = nn.Conv3d(16, 32, [3, 1, 4], stride=1, padding=0)
        self.conv21_bn = nn.BatchNorm2d(32)
        self.prelu21 = nn.PReLU()
        self.conv22 = nn.Conv3d(32, 32, [3, 8, 1], stride=(1, 2, 1), padding=0)
        self.conv22_bn = nn.BatchNorm2d(32)
        self.prelu22 = nn.PReLU()
        #### conv-3 ####
        self.conv31 = nn.Conv3d(32, 64, [1, 1, 3], stride=1, padding=0)
        self.conv31_bn = nn.BatchNorm2d(64)
        self.prelu31 = nn.PReLU()
        self.conv32 = nn.Conv3d(64, 64, [1, 7, 1], stride=1, padding=0)
        self.conv32_bn = nn.BatchNorm2d(64)
        self.prelu32 = nn.PReLU()
        #### conv-4 ####
        self.conv41 = nn.Conv3d(64, 128, [1, 1, 3], stride=1, padding=0)
        self.conv41_bn = nn.BatchNorm2d(128)
        self.prelu41 = nn.PReLU()
        self.conv42 = nn.Conv3d(128, 128, [1, 7, 1], stride=1, padding=0)
        self.conv42_bn = nn.BatchNorm2d(128)
        self.prelu42 = nn.PReLU()
        #### conv-5 ####
        self.fc5 = nn.Conv3d(128, 128, [1, 3, 3], stride=1, padding=0)
        self.prelu5 = nn.PReLU()

        self.last_fc = nn.Linear(128, nb_class)



    def forward(self, x):
        # x: (batch_size, C, nb_uttrs, H, W) = (batch_size, channels, uttrs, freq, windows)
        if x.dim() == 4:
            x = torch.unsqueeze(x, 1)

        net = self.prelu11(self.conv11_bn(self.conv11(x)))
        net = F.max_pool3d(self.prelu12((self.conv12_bn(self.conv12(net)))), [1, 1, 2], [1, 1, 2])

        net = self.prelu21(self.conv21_bn(self.conv21(net)))
        net = F.max_pool3d(self.prelu22((self.conv22_bn(self.conv22(net)))), [1, 1, 2], [1, 1, 2])

        net = self.prelu31(self.conv31_bn(self.conv31(net)))
        net = self.prelu32(self.conv32_bn(self.conv32(net)))

        net = self.prelu41(self.conv41_bn(self.conv41(net)))
        net = self.prelu42(self.conv42_bn(self.conv42(net)))

        net = self.prelu5((self.fc5(net)))
        net = net.view(-1, num_flat_features(net))
        net = self.last_fc(net)

        return net
