import torch
import torch.nn as nn
import math

from .model import AngleLinear


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

class st_pool_layer_2d(nn.Module):
    def __init__(self):
        super(st_pool_layer_2d, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        mean = x.mean(2)
        std = x.std(2)
        stat = torch.cat([mean, std], -1)

        return stat

class tdnn_xvector(nn.Module):
    """xvector architecture"""
    def __init__(self, config, base_width=512,  n_labels=31):
        super(tdnn_xvector, self).__init__()
        inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, base_width, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, 1500, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(1500),
            nn.ReLU(True),
            st_pool_layer(),
            nn.Linear(3000, base_width),
            nn.BatchNorm1d(base_width),
        )

        loss_type = config["loss"]
        if loss_type == "angular":
            last_fc = AngleLinear(base_width, n_labels)
        elif loss_type == "softmax":
            last_fc = nn.Linear(base_width, n_labels)
        else:
            print("not implemented loss")
            raise NotImplementedError

        self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(base_width, base_width),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            last_fc
        )


        self._initialize_weights()

    def load_extractor(self, state_dict):
        state_dict.pop("classifier.4.weight")
        state_dict.pop("classifier.4.bias")
        self.load_state_dict(state_dict)

    def embed(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
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

from model.tdnnModel import st_pool_layer

class tdnn_xvector_center(nn.Module):
    """xvector architecture
        tdnn6.affine is embeding layer no
        untying classifier for flexible embedding positon
    """
    def __init__(self, config, base_width, n_labels=31):
        super(tdnn_xvector_center, self).__init__()
        inDim = 30
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, base_width, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            nn.Conv1d(base_width, 1500, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(1500),
            nn.ReLU(True),
            st_pool_layer(),
            nn.Linear(3000, base_width),
        )

        loss_type = config["loss"]
        if loss_type == "angular":
            last_fc = AngleLinear(base_width, n_labels)
        elif loss_type == "softmax":
            last_fc = nn.Linear(base_width, n_labels)
        else:
            print("not implemented loss")
            raise NotImplementedError

        self.tdnn6_bn = nn.BatchNorm1d(base_width)
        self.tdnn6_relu = nn.ReLU(True)
        self.tdnn7_affine = nn.Linear(base_width, base_width)
        self.tdnn7_bn = nn.BatchNorm1d(base_width)
        self.tdnn7_relu = nn.ReLU(True)
        self.tdnn8_last = last_fc


        self._initialize_weights()

    def embed(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn(x)

        return x

    def feat_out(self, x):
        x = self.embed(x)
        feat = x
        x = self.tdnn6_bn(x)
        x = self.tdnn6_relu(x)
        x = self.tdnn7_affine(x)
        x = self.tdnn7_bn(x)
        x = self.tdnn7_relu(x)
        x = self.tdnn8_last(x)

        return feat, x
    
    def forward(self, x):
        x = self.embed(x)
        x = self.tdnn6_bn(x)
        x = self.tdnn6_relu(x)
        x = self.tdnn7_affine(x)
        x = self.tdnn7_bn(x)
        x = self.tdnn7_relu(x)
        x = self.tdnn8_last(x)

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


class permute_dim(nn.Module):
    def __init__(self):
        super(permute_dim, self).__init__()

    def forward(self, x):

        # x = x.permute(0,2,1)
        return x.permute(0,2,1)

                

class tdnn_xvector_dense(tdnn_xvector):
    """xvector architecture"""
    def __init__(self, config, n_labels=31):
        super(tdnn_xvector_dense, self).__init__(config, n_labels)
        inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, 512, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=1, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, stride=1, dilation=1, kernel_size=3),
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

        loss_type = config["loss"]
        if loss_type == "angular":
            last_fc = AngleLinear(512, n_labels)
        elif loss_type == "softmax":
            last_fc = nn.Linear(512, n_labels)
        else:
            print("not implemented loss")
            raise NotImplementedError

        self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            last_fc
        )


        self._initialize_weights()

class tdnn_xvector_2d(nn.Module):
    """xvector architecture
        tdnn6.affine is embeding layer no
        untying classifier for flexible embedding positon
        conv1d --> conv2d
    """
    def __init__(self, config, base_width, n_labels=31):
        super(tdnn_xvector_2d, self).__init__()
        # inDim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv2d(1, base_width, stride=1, dilation=1, kernel_size=5),
            nn.BatchNorm2d(base_width),
            nn.ReLU(True),
            nn.Conv2d(base_width, base_width, stride=1, dilation=3, kernel_size=3),
            nn.BatchNorm2d(base_width),
            nn.ReLU(True),
            nn.Conv2d(base_width, base_width, stride=1, dilation=4, kernel_size=3),
            nn.BatchNorm2d(base_width),
            nn.ReLU(True),
            nn.Conv2d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(True),
            nn.Conv2d(base_width, 1024, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            st_pool_layer_2d(),
            nn.Linear(2048, base_width),
        )

        loss_type = config["loss"]
        if loss_type == "angular":
            last_fc = AngleLinear(base_width, n_labels)
        elif loss_type == "softmax":
            last_fc = nn.Linear(base_width, n_labels)
        else:
            print("not implemented loss")
            raise NotImplementedError

        self.tdnn6_bn = nn.BatchNorm1d(base_width)
        self.tdnn6_relu = nn.ReLU(True)
        self.tdnn7_affine = nn.Linear(base_width, base_width)
        self.tdnn7_bn = nn.BatchNorm1d(base_width)
        self.tdnn7_relu = nn.ReLU(True)
        self.tdnn8_last = last_fc


        self._initialize_weights()

    def embed(self, x):
        # x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        # x = x.permute(0,2,1)
        x = self.tdnn(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.tdnn6_bn(x)
        x = self.tdnn6_relu(x)
        x = self.tdnn7_affine(x)
        x = self.tdnn7_bn(x)
        x = self.tdnn7_relu(x)
        x = self.tdnn8_last(x)

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


class permute_dim(nn.Module):
    def __init__(self):
        super(permute_dim, self).__init__()

    def forward(self, x):

        # x = x.permute(0,2,1)
        return x.permute(0,2,1)


class tdnn_xvector_cross(nn.Module):
    """xvector architecture
        tdnn6.affine is embeding layer no
        untying classifier for flexible embedding positon
        conv1d --> conv2d
    """
    def __init__(self, config, base_width, n_labels=31):
        super(tdnn_xvector_cross, self).__init__()
        inDim = config['input_dim']
        spFr =  config['splice_frames'][-1]
        self.tdnn = nn.Sequential(
            nn.Conv1d(inDim, base_width, stride=1, dilation=1, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            permute_dim(),
            nn.Conv1d(spFr, base_width, stride=1, dilation=3, kernel_size=3, padding=3),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            permute_dim(),
            nn.Conv1d(base_width, base_width, stride=1, dilation=4, kernel_size=3, padding=4),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            permute_dim(),
            nn.Conv1d(base_width, base_width, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(base_width),
            nn.ReLU(True),
            permute_dim(),
            nn.Conv1d(base_width, 1500, stride=1, dilation=1, kernel_size=1),
            nn.BatchNorm1d(1500),
            nn.ReLU(True),
            st_pool_layer(),
            nn.Linear(3000, base_width),
            nn.BatchNorm1d(base_width),
        )

        loss_type = config["loss"]
        if loss_type == "angular":
            last_fc = AngleLinear(base_width, n_labels)
        elif loss_type == "softmax":
            last_fc = nn.Linear(base_width, n_labels)
        else:
            print("not implemented loss")
            raise NotImplementedError

        self.tdnn6_bn = nn.BatchNorm1d(base_width)
        self.tdnn6_relu = nn.ReLU(True)
        self.tdnn7_affine = nn.Linear(base_width, base_width)
        self.tdnn7_bn = nn.BatchNorm1d(base_width)
        self.tdnn7_relu = nn.ReLU(True)
        self.tdnn8_last = last_fc


        self._initialize_weights()

    def embed(self, x):
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.tdnn6_bn(x)
        x = self.tdnn6_relu(x)
        x = self.tdnn7_affine(x)
        x = self.tdnn7_bn(x)
        x = self.tdnn7_relu(x)
        x = self.tdnn8_last(x)

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


if __name__ == '__main__':
    config = {'loss':'softmax', 'input_dim':30,
            'splice_frames':100}
    net=tdnn_xvector_cross(config, 512, 10)
    print(net)
    y = net(torch.randn(15, 100, 30))
