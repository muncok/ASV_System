import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from .model import SerializableModule, num_flat_features
from .AuxModels import conv_block

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

class TdnnCNN(SerializableModule):
    def __init__(self, config, n_labels, embed_mode=False):
        super().__init__()
        self.splice_frames = config["splice_frames"]
        hid_dim = 64
        self.feat_size = 64
        self.convb_1 = conv_block(1, hid_dim)
        self.convb_2 = conv_block(hid_dim, hid_dim)
        self.convb_3 = conv_block(hid_dim, hid_dim)
        if self.splice_frames < 21:
            self.convb_4 = conv_block(hid_dim, hid_dim, 1)
        else:
            self.convb_4 = conv_block(hid_dim, hid_dim)

        with torch.no_grad():
            test_in = torch.zeros((1, 1, self.splice_frames, 40))
            test_out = self.embed(test_in)
            self.feat_dim = test_out.size(-1)
            if not embed_mode:
                self.output = nn.Linear(self.feat_dim, n_labels)
        self.embed_mode = embed_mode

    def embed(self, seq_x):
        # input is full sequence, not a snippet
        embeds = []
        if seq_x.dim() == 3:
            seq_x = seq_x.unsqueeze(1)
        for i in range(0, seq_x.size(2) - self.splice_frames+1, 1):
            x = seq_x.narrow(2, i, self.splice_frames)
            x = self.convb_1(x)
            x = self.convb_2(x)
            x = self.convb_3(x)
            x = self.convb_4(x)
            x = x.view(-1, num_flat_features(x))
            embeds.append(x)
        x = torch.stack(embeds, dim=1)  # consistency for TDNN layer
        return x

    def forward(self, x):
        x = self.embed(x)
        if not self.embed_mode:
            x = self.output(x)
        return x

class TdnnStatCNN(SerializableModule):
    def __init__(self, config, n_labels):
        super().__init__()
        self.splice_frames = config["splice_frames"]
        self.stride_frames = config["stride_frames"]
        hid_dim = 64
        self.feat_size = 64
        self.convb_1 = conv_block(1, hid_dim)
        self.convb_2 = conv_block(hid_dim, hid_dim)
        self.convb_3 = conv_block(hid_dim, hid_dim)
        if self.splice_frames < 21:
            self.convb_4 = conv_block(hid_dim, hid_dim, 1)
        else:
            self.convb_4 = conv_block(hid_dim, hid_dim)

        with torch.no_grad():
            test_in = torch.zeros((1, 1, self.splice_frames, 40))
            test_out = self.embed(test_in)
            self.feat_dim = test_out.size(-1)

    def embed(self, seq_x):
        # input is full sequence, not a snippet
        embeds = []
        if seq_x.dim() == 3:
            seq_x = seq_x.unsqueeze(1)
        for i in range(0, seq_x.size(2) - self.splice_frames+1, self.stride_frames):
            x = seq_x.narrow(2, i, self.splice_frames)
            x = self.convb_1(x)
            x = self.convb_2(x)
            x = self.convb_3(x)
            x = self.convb_4(x)
            x = x.view(-1, num_flat_features(x))
            embeds.append(x)
        x = torch.stack(embeds, dim=1)  # consistency for TDNN layer
        return x

    def forward(self, x):
        x = self.embed(x)
        return x

class TdnnModel(SerializableModule):
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
            test_in = torch.zeros((1, 1, config['splice_frames']+4+8, 40))
            test_out = self.embed(test_in)
            out_feat_dim = test_out.size(-1)
        self.output = nn.Linear(out_feat_dim, n_labels)

    def embed(self, x):
        x = self.extractor(x)
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

class TdnnStatModel(SerializableModule):
    def __init__(self, config, n_labels, embed_mode=False):
        super().__init__()
        self.embed_mode = embed_mode
        # [-4, +4] 9 frames
        self.extractor = TdnnStatCNN(config, n_labels)
        feat_dim = self.extractor.feat_dim
        self.tdnn1 = TDNN([-2, 2], input_dim=feat_dim, output_dim=512, full_context=True)
        self.tdnn2 = TDNN([0, 0], input_dim=512, output_dim=1024, full_context=True)
        self.tdnn3 = TDNN([-4, 4], input_dim=1024, output_dim=1024, full_context=True)
        self.tdnn4 = TDNN([0, 0], input_dim=1024, output_dim=1024, full_context=True)
        self.fc1 = nn.Linear(1024*2, 256)
        self.output = nn.Linear(256, n_labels)

    def embed(self, x):
        #print("x_shape: {}".format(x.shape))
        x = self.extractor(x)
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        mean = x.mean(1)
        std = x.std(1)
        stat = torch.cat([mean, std], -1)
        x = self.fc1(stat)
        # print(stat.shape)
        return x

    def forward(self, x):
        x = self.embed(x)
        if not self.embed_mode:
            x = self.output(x)
        return x
