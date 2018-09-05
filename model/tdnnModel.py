# This codes based on  https://github.com/SiddGururani/Pytorch-TDNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from .model import SerializableModule, num_flat_features
from .model import AngleLinear

"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""

class TDNN(nn.Module):
    def __init__(self, context, input_dim, output_dim, full_context = True):
        """
        Definition of context is the same as the way it's defined in the Peddinti
        paper. It's a list of integers, eg: [-2,2]
        By deault, full context is chosen, which means: [-2,2] will be expanded to
        [-2,-1,0,1,2] i.e. range(-2,3)
        """
        super(TDNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.check_valid_context(context)
        self.kernel_width, context = self.get_kernel_width(context,full_context)  # return len(context), context
        self.register_buffer('context',torch.LongTensor(context))
        self.full_context = full_context
        stdv = 1./math.sqrt(input_dim)
        self.kernel = nn.Parameter(torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(0,stdv))
        self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0,stdv))
        # self.cuda_flag = False

    def forward(self,x):
        """
        x is one batch of data
        x.size(): [batch_size, sequence_length, input_dim]
        sequence length is the length of the input spectral data (number of frames) or
        if already passed through the convolutional network,
        it's the number of learned features

        output size: [batch_size, output_dim, len(valid_steps)]
        """
        # Check if parameters are cuda type and change context
        # if type(self.bias.data) == torch.cuda.FloatTensor and self.cuda_flag == False:
        #     self.context = self.context.cuda()
        #     self.cuda_flag = True
        conv_out = self.special_convolution(x, self.kernel, self.context, self.bias)
        return F.relu(conv_out)

    def special_convolution(self, x, kernel, context, bias):
        """
        This function performs the weight multiplication given an arbitrary context.
        Cannot directly use convolution because in case of only particular frames of
        context,
        one needs to select only those frames and perform a convolution across all
        batch items and all output dimensions of the kernel.
        """
        input_size = x.size()
        assert len(input_size) == 3, 'Input tensor dimensionality is incorrect. \
        Should be a 3D tensor'
        [batch_size, input_sequence_length, input_dim] = input_size
        x = x.transpose(1,2).contiguous()

        # Allocate memory for output
        valid_steps = self.get_valid_steps(self.context, input_sequence_length)
        xs = Variable(self.bias.data.new(batch_size, kernel.size()[0],
            len(valid_steps)))

        # Perform the convolution with relevant input frames
        for c, i in enumerate(valid_steps):
            features = torch.index_select(x, 2, context+i)
            xs[:, :, c] = F.conv1d(features, kernel, bias=bias)[:, :, 0]
        xs = xs.transpose(1,2).contiguous()
        return xs

    @staticmethod
    def check_valid_context(context):
        # here context is still a list
        assert context[0] <= context[-1], 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0],context[-1]+1)
        return len(context), context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        start = 0 if context[0] >= 0 else -1*context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        return range(start, end)


def statistic_pool(x):
    mean = x.mean(1)
    std = x.std(1)
    stat = torch.cat([mean, std], -1)
    return stat

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

class TdnnCNN(SerializableModule):
    def __init__(self, config, n_labels, embed_mode=False):
        super().__init__()
        self.input_frames = config["input_frames"]
        self.splice_frames = config["splice_frames"]
        self.stride_frames = config["stride_frames"]
        hid_dim = 64
        self.convb_1 = conv_block(1, hid_dim)
        self.convb_2 = conv_block(hid_dim, hid_dim)
        self.convb_3 = conv_block(hid_dim, hid_dim)
        if self.splice_frames < 21:
            self.convb_4 = conv_block(hid_dim, hid_dim, 1)
        else:
            self.convb_4 = conv_block(hid_dim, hid_dim)

        input_dim = config['input_dim']
        with torch.no_grad():
            test_in = torch.zeros((1, 1, self.input_frames, input_dim))
            test_out = self.forward(test_in)
            self.feat_dim = test_out.size(-1)

    def forward(self, seq_x):
        # input is full sequence, not a snippet
        embeds = []
        if seq_x.dim() == 3:
            seq_x = seq_x.unsqueeze(1)
        for i in range(0, seq_x.size(2)-self.splice_frames+1, self.stride_frames):
            x = seq_x.narrow(2, i, self.splice_frames)
            x = self.convb_1(x)
            x = self.convb_2(x)
            x = self.convb_3(x)
            x = self.convb_4(x)
            x = x.view(-1, num_flat_features(x))
            embeds.append(x)
        x = torch.stack(embeds, dim=1)  # consistency for TDNN layer

        return x


class CTdnnModel(SerializableModule):
    def __init__(self, config, n_labels, embed_mode=False):
        super().__init__()
        self.embed_mode = embed_mode
        # [-4, +4] 9 frames
        self.extractor = TdnnCNN(config, n_labels, embed_mode=True)
        feat_dim = self.extractor.feat_dim
        self.tdnn1 = TDNN([-2, 2], input_dim=feat_dim, output_dim=512, full_context=True)
        self.tdnn2 = TDNN([0, 0], input_dim=512, output_dim=1024, full_context=True)
        self.tdnn3 = TDNN([-4, 4], input_dim=1024, output_dim=1024, full_context=True)
        self.tdnn4 = TDNN([0, 0], input_dim=1024, output_dim=1024, full_context=True)
        with torch.no_grad():
            seq_len = config['input_frames']
            input_dim = config['input_dim']
            test_in = torch.zeros((1, 1, seq_len, input_dim))
            test_out = self.embed(test_in)
            out_feat_dim = test_out.size(-1)
        self.output = nn.Linear(out_feat_dim, n_labels)

    def embed(self, x):
        x = self.extractor(x)
        # input: (batch_size, seq, input_dim)
        # print(x.shape)
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = x.view(-1, num_flat_features(x))
        return x

    def forward(self, x):
        x = self.embed(x)
        if not self.embed_mode:
            x = self.output(x)
        return x


class TdnnModel(SerializableModule):
    def __init__(self, config, n_labels):
        super().__init__()
        input_dim_ = config['input_dim']
        self.tdnn1 = TDNN([-2, 2], input_dim=input_dim_, output_dim=512,
                full_context=True)
        self.tdnn2 = TDNN([-1, 1], input_dim=512, output_dim=512,
                full_context=True)
        self.tdnn3 = TDNN([-1, 1], input_dim=512, output_dim=512,
                full_context=True)
        self.tdnn4 = TDNN([0, 0], input_dim=512, output_dim=512,
                full_context=True)
        self.tdnn5 = TDNN([0, 0], input_dim=512, output_dim=1500,
                full_context=True)
        self.fc1 = nn.Linear(1500*2, 512)
        self.fc2 = nn.Linear(512, 300)
        loss_type = config.get("loss", "")
        if loss_type == "angle":
            self.output = AngleLinear(300, n_labels)
        else:
            self.output = nn.Linear(300, n_labels)

    def embed(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        # print("x_shape: {}".format(x.shape))
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = statistic_pool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.output(x)
        return x


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
        x = x.squeeze(1)
        # (batch, time, freq) -> (batch, freq, time)
        x = x.permute(0,2,1)
        x = self.tdnn(x)

        return x

    def forward(self, x):
        x = self.embed(x)
        x = self.classifier(x)

        return x

class tdnn_xvector_v1(gTDNN):
    """xvector architecture"""
    def __init__(self, config, n_labels=31):
        super(tdnn_xvector_v1, self).__init__(config, n_labels)
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
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, n_labels),
        )
        self._initialize_weights()

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

