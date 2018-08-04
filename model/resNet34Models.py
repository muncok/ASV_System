import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from .model import AngleLinear

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
        if loss_type == "angular":
            self.output = AngleLinear(128, n_labels)
        elif loss_type == "softmax":
            self.output = nn.Linear(128, n_labels)
        else:
            raise NotImplementedError

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

class ScaleResNet34(ResNet34):
    def __init__(self, config, layers, n_labels=1000, alpha=12):
        super().__init__(config, layers, n_labels)
        self.alpha = alpha

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

        x = x / x.norm(2, dim=1, keepdim=True)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.alpha * x
        x = self.output(x)

        return x

class ResNet34_v1(ResNet34):
    def __init__(self, config, layers, n_labels=1000):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 16, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, layers[3], stride=2)
        self.fc = nn.Linear(128 * BasicBlock.expansion, 128)
        loss_type = config["loss"]
        if loss_type == "angular":
            self.output = AngleLinear(128, n_labels)
        elif loss_type == "softmax":
            self.output = nn.Linear(128, n_labels)
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

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

class ResNet34_v2(ResNet34):
    """
        remove maxpooling and fc
        change first conv's kernel_size 7 --> 3
    """
    def __init__(self, config, layers, n_labels=1000):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 16, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, layers[3], stride=2)
        loss_type = config["loss"]
        if loss_type == "angular":
            self.output = AngleLinear(128, n_labels)
        elif loss_type == "softmax":
            self.output = nn.Linear(128, n_labels)
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,x.shape[-2:])
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.output(x)

        return x

class ResNet34_v3(ResNet34):
    """
        remove maxpooling and keep fc
        change first conv's kernel_size 7 --> 3
    """
    def __init__(self, config, layers, n_labels=1000):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 16, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 128, layers[3], stride=2)
        self.fc = nn.Linear(128 * BasicBlock.expansion, 128)
        loss_type = config["loss"]
        if loss_type == "angular":
            self.output = AngleLinear(128, n_labels)
        elif loss_type == "softmax":
            self.output = nn.Linear(128, n_labels)
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

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

class ResNet34_v3_w(ResNet34):
    """
        remove maxpooling and keep fc
        change first conv's kernel_size 7 --> 3
        double the all feature maps
    """
    def __init__(self, config, layers, n_labels=1000):
        super(ResNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, 32, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, layers[3], stride=2)
        self.fc = nn.Linear(256 * BasicBlock.expansion, 512)
        loss_type = config["loss"]
        if loss_type == "angular":
            self.output = AngleLinear(512, n_labels)
        elif loss_type == "softmax":
            self.output = nn.Linear(512, n_labels)
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,x.shape[-2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.output(x)

        return x


class ResNet34_v4(ResNet34_v3):
    """
        remove maxpooling
        change first conv's kernel_size 7 --> 3
        add relu for fc
    """
    def __init__(self, config, layers, n_labels=1000):
        super(ResNet34_v4, self).__init__(config, layers, n_labels)

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,x.shape[-2:])
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = F.relu(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.output(x)

        return x


class ScaleResNet34_v4(ResNet34_v4):
    def __init__(self, config, layers, n_labels=1000, alpha=12.0):
        super().__init__(config, layers, n_labels)
        self.alpha = alpha

    def embed(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x,x.shape[-2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.relu(x)

        # x = x / x.norm(2, dim=1, keepdim=True)
        x = F.normalize(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.alpha * x
        x = self.output(x)

        return x

class ResNet50(ResNet):
    def __init__(self, config, layers = [3, 4, 6, 3], n_labels=1000):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 16, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 128, layers[3], stride=2)
        self.fc = nn.Linear(128 * Bottleneck.expansion, 128)
        loss_type = config["loss"]
        if loss_type == "angular":
            self.output = AngleLinear(128, n_labels)
        elif loss_type == "softmax":
            self.output = nn.Linear(128, n_labels)
        else:
            raise NotImplementedError

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

