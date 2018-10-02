import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .ResNet34 import ResNet34

# from .model import AngleLinear


# coding=utf-8
# Copyright 2018 jose.fonollosa@upc.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class st_pool_layer(nn.Module):
    def __init__(self):
        super(st_pool_layer, self).__init__()

    def forward(self, x):
        mean = x.mean(2)
        std = x.std(2)
        stat = torch.cat([mean, std], -1)

        return stat

class gTDNN(nn.Module):
    def __init__(self, config, n_labels=31):
        super(gTDNN, self).__init__()
        inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, 450, stride=1, dilation=1, kernel_size=3),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=1, kernel_size=4),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.Conv1d(450, 450, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool1d(3, stride=3),
            st_pool_layer(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(900, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, n_labels),
        )
        self._initialize_weights()

    def embed(self, x):
        x = x.squeeze()
        x = x.permute(0,2,1)
        x = self.tdnn(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class tdnn_xvector(gTDNN):
    """xvector architecture"""
    def __init__(self, config, n_labels=31):
        super(tdnn_xvector, self).__init__(config, n_labels)
        inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, 512, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 1500, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(1500),
            nn.ReLU(True),
            st_pool_layer(),
            nn.Linear(3000, 512),
            nn.BatchNorm1d(512),
        )
        self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, n_labels),
        )
        self._initialize_weights()

    def embed(self, x):
        x = x.squeeze()
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.classifier(x)

        return x

class tdnn_conv(gTDNN):
    """xvector architecture"""
    def __init__(self, config, n_labels=31):
        super(tdnn_conv, self).__init__(config, n_labels=n_labels)
        inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, 512, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )

        from .ResNet34 import BasicBlock
        layers = [3,4,6,3]
        self.inplanes = inplanes = 16
        self.extractor = nn.Sequential(
            nn.Conv2d(1, inplanes, kernel_size=3, stride=1, padding=3,
                                   bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, inplanes, layers[0]),
            self._make_layer(BasicBlock, 2*inplanes, layers[1], stride=2),
            self._make_layer(BasicBlock, 4*inplanes, layers[2], stride=2),
            self._make_layer(BasicBlock, 8*inplanes, layers[3], stride=2)
        )

        self.classifier = nn.Linear(8*inplanes, n_labels)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            # classifier does not contain Conv2d or BN2d
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def embed(self, x):
        x = x.squeeze()
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn(x)
        x = self.extractor(x)
        x = F.avg_pool2d(x,x.shape[-2:])
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.classifier(x)

        return x

class tdnn_2dim(gTDNN):
    """xvector architecture"""
    def __init__(self, config, n_labels=31):
        super(tdnn_2dim, self).__init__(config, n_labels)
        inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, 128, stride=1, dilation=1, kernel_size=5),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, stride=1, dilation=3, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            st_pool_layer(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 2),
            nn.ReLU(True),
            nn.BatchNorm1d(2),
            nn.Linear(2, n_labels),
        )
        self._initialize_weights()

    def embed(self, x):
        x = x.squeeze()
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.classifier(x)

        return x
