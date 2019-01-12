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

import torch
import torch.nn as nn
import math

class st_pool_layer(nn.Module):
    def __init__(self):
        super(st_pool_layer, self).__init__()

    def forward(self, x):
        mean = x.mean(2)
        std = x.std(2)
        stat = torch.cat([mean, std], -1)

        return stat

class tdnn_xvector(nn.Module):
    """xvector architecture"""
    def __init__(self, config, base_width, n_labels):
        super(tdnn_xvector, self).__init__()
        in_dim = config['input_dim']
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_dim, base_width, stride=1, dilation=1, kernel_size=5),
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

        last_fc = nn.Linear(base_width, n_labels)

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

if __name__ == '__main__':
    config = {'loss':'softmax', 'input_dim':30,
            'splice_frames':100}
    net=tdnn_xvector(config, 512, 10)
    print(net)
    y = net(torch.randn(15, 100, 30))
