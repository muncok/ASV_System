from enum import Enum

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class ConfigType(Enum):
    CNN_TRAD_POOL2 = "cnn-trad-pool2" # default full model (TF variant)
    CNN_ONE_STRIDE1 = "cnn-one-stride1" # default compact model (TF variant)
    CNN_ONE_FPOOL3 = "cnn-one-fpool3"
    CNN_ONE_FSTRIDE4 = "cnn-one-fstride4"
    CNN_ONE_FSTRIDE8 = "cnn-one-fstride8"
    CNN_TPOOL2 = "cnn-tpool2"
    CNN_TPOOL3 = "cnn-tpool3"
    CNN_TSTRIDE2 = "cnn-tstride2"
    CNN_TSTRIDE4 = "cnn-tstride4"
    CNN_TSTRIDE8 = "cnn-tstride8"
    RES15 = "res15"
    RES26 = "res26"
    RES8 = "res8"
    RES15_NARROW = "res15-narrow"
    RES15_WIDE = "res15-wide"
    RES8_NARROW = "res8-narrow"
    RES8_WIDE = "res8-wide"
    RES26_NARROW = "res26-narrow"
    LSTM = "lstm"
    CNN_SMALL = "cnn-small"
    CNN_FRAMES = "cnn-frames"

def find_model(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    if conf.startswith("res"):
        return SpeechResModel
    elif conf.startswith("cnn"):
        return SpeechModel
    else:
        return SpeechLSTMModel

def find_config(conf):
    if isinstance(conf, ConfigType):
        conf = conf.value
    return _configs[conf]

def truncated_normal(tensor, std_dev=0.01):
    tensor.zero_()
    tensor.normal_(std=std_dev)
    while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
        t = tensor[torch.abs(tensor) > 2 * std_dev]
        t.zero_()
        tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))

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
        self.fc6 = nn.Conv2d(256, 4096, [9, 1])
        self.fc6_bn = nn.BatchNorm2d(4096)
        self.fc7 = nn.Linear(4096, 1024)
        self.fc7_bn = nn.BatchNorm2d(1024)
        self.fc8 = nn.Linear(1024, nb_class)

        self.feat_size = 1024

    def embed(self, x):
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
        return fc7_out

    def forward(self, x):
        x = self.embed(x)
        fc8_out = self.fc8(x)
        return fc8_out


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
        self.feat_size = self.output.in_features

    def embed(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
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
        return x

    def forward(self, x):
        x = self.embed(x)
        return self.output(x)

class SpeechLSTMModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        target_size = config["n_labels"]
        no_cuda = config['no_cuda']
        self.input_dim = config["in_dim"]
        self.hidden_dim = config["h_dim"]
        self.n_layers = config["n_layers"]
        self.batch_size = config["batch_size"]

        self.lstm = nn.LSTM(input_size=self.input_dim , hidden_size=self.hidden_dim, num_layers=self.n_layers)
        self.proj = nn.Linear(self.hidden_dim, target_size)
        self.hidden = self.init_hidden(no_cuda=no_cuda)
        self.feat_size = self.hidden_dim

    def init_hidden(self, no_cuda=False):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        hidden_init = Variable(nn.init.uniform(torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                                               a=-0.02, b=0.02))
        cell_init = Variable(nn.init.uniform(torch.zeros(self.n_layers, self.batch_size, self.hidden_dim),
                                                a=-0.02, b=0.02))
        if not no_cuda:
            hidden_init = hidden_init.cuda()
            cell_init = cell_init.cuda()
        return (hidden_init, cell_init)

    def embed(self, x):
        # input: (sequence, batch, features)

        # hidden = self.hidden
        # x = x.contiguous()
        # for i in x:
        # # Step through the sequence one element at a time.
        # # after each step, hidden contains the hidden state.
        # i = i.view(1, self.batch_size, -1)
        # out, hidden = self.lstm(i, hidden)

        lstm_out, hidden = self.lstm(x, self.hidden)
        last_out = lstm_out[-1]
        return last_out

    def forward(self, x):
        last_out = self.embed(x)
        proj_out = self.proj(last_out)
        return proj_out

class SpeechModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        n_labels = config["n_labels"]
        n_featmaps1 = config["n_feature_maps1"]

        conv1_size = config["conv1_size"] # (time, frequency)
        conv1_pool = config["conv1_pool"]
        conv1_stride = tuple(config["conv1_stride"])
        dropout_prob = config["dropout_prob"]
        width = config["width"]
        height = config["height"]
        self.conv1 = nn.Conv2d(1, n_featmaps1, conv1_size, stride=conv1_stride)
        tf_variant = config.get("tf_variant")
        self.tf_variant = tf_variant
        if tf_variant:
            truncated_normal(self.conv1.weight.data)
            self.conv1.bias.data.zero_()
        self.pool1 = nn.MaxPool2d(conv1_pool)

        x = Variable(torch.zeros(1, 1, height, width), volatile=True)
        x = self.pool1(self.conv1(x))
        conv_net_size = x.view(1, -1).size(1)
        last_size = conv_net_size

        if "conv2_size" in config:
            conv2_size = config["conv2_size"]
            conv2_pool = config["conv2_pool"]
            conv2_stride = tuple(config["conv2_stride"])
            n_featmaps2 = config["n_feature_maps2"]
            self.conv2 = nn.Conv2d(n_featmaps1, n_featmaps2, conv2_size, stride=conv2_stride)
            if tf_variant:
                truncated_normal(self.conv2.weight.data)
                self.conv2.bias.data.zero_()
            self.pool2 = nn.MaxPool2d(conv2_pool)
            x = self.pool2(self.conv2(x))
            conv_net_size = x.view(1, -1).size(1)
            last_size = conv_net_size
        if not tf_variant:
            self.lin = nn.Linear(conv_net_size, 32)

        if "dnn1_size" in config:
            dnn1_size = config["dnn1_size"]
            last_size = dnn1_size
            if tf_variant:
                self.dnn1 = nn.Linear(conv_net_size, dnn1_size)
                truncated_normal(self.dnn1.weight.data)
                self.dnn1.bias.data.zero_()
            else:
                self.dnn1 = nn.Linear(32, dnn1_size)
            if "dnn2_size" in config:
                dnn2_size = config["dnn2_size"]
                last_size = dnn2_size
                self.dnn2 = nn.Linear(dnn1_size, dnn2_size)
                if tf_variant:
                    truncated_normal(self.dnn2.weight.data)
                    self.dnn2.bias.data.zero_()
        if "bn_size" in config:
            self.bn_size = config["bn_size"]
            self.bottleneck = nn.Linear(last_size, self.bn_size)
            last_size = self.bn_size

        self.output = nn.Linear(last_size, n_labels)
        if tf_variant:
            truncated_normal(self.output.weight.data)
            self.output.bias.data.zero_()
        self.dropout = nn.Dropout(dropout_prob)

        self.feat_size = self.output.in_features

    def embed(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1))) # shape: (batch, channels, i1, o1)
        x = self.dropout(x)
        x = self.pool1(x)
        if hasattr(self, "conv2"):
            x = F.relu(self.conv2(x)) # shape: (batch, o1, i2, o2)
            x = self.dropout(x)
            x = self.pool2(x)
        x = x.view(x.size(0), -1) # shape: (batch, o3)
        if hasattr(self, "lin"):
            x = self.lin(x)
        if hasattr(self, "dnn1"):
            x = self.dnn1(x)
            if not self.tf_variant:
                x = F.relu(x)
            x = self.dropout(x)
        if hasattr(self, "dnn2"):
            x = self.dnn2(x)
            x = self.dropout(x)
        if hasattr(self, "bn_size"):
            x = self.bottleneck(x)

    def forward(self, x):
        return self.output(x)

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class SimpleCNN(SerializableModule):
    def __init__(self):
        super().__init__()
        hid_dim = 64
        self.feat_size = 64
        self.convb_1 = conv_block(1, hid_dim)
        self.convb_2 = conv_block(hid_dim, hid_dim)
        self.convb_3 = conv_block(hid_dim, hid_dim)
        self.convb_4 = conv_block(hid_dim, hid_dim)

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
        if hasattr(self, "output"):
            x = self.output(x)
        return x
        # if feature:
        #     return x
        # else:
        #     x = self.output(x)
        #     return x


class MTLSpeechModel(SpeechModel):
    def __init__(self, config):
        super().__init__(config)
        n_labels1 = config["n_labels1"]
        bn_size = config['bn_size']
        self.output1 = nn.Linear(bn_size, n_labels1)

    def forward(self, x, task=0):
        x = super().forward(x, feature=True)
        if task != 0:
            return self.output1(x)
        else:
            return self.output(x)


class logRegModel(SpeechModel):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(1,1)

    def forward(self, x, spk_model=None, feature=False):
        x = super().forward(x, feature=True)
        spk_model = spk_model.expand(x.size(0), x.size(1))
        x = F.cosine_similarity(x, spk_model)
        x = x.unsqueeze(1)
        x = self.linear(x)
        return x


class GatedCNN(SerializableModule):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 seq_len,
                 embd_size,
                 n_layers,
                 kernel,
                 out_chs,
                 res_block_count,
                 ans_size):
        super(GatedCNN, self).__init__()
        self.res_block_count = res_block_count
        self.limit = n_layers

        # self.embedding = nn.Embedding(vocab_size, embd_size)

        # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        time_pad= kernel[0]//2
        self.conv_0 = nn.Conv2d(1, out_chs, kernel, padding=(time_pad, 0))
        self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel, padding=(time_pad, 0))
        self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))

        self.conv = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1),
                                             padding=(time_pad, 0)) for _ in range(n_layers)])
        self.conv_gate = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1),
                                                  padding=(time_pad, 0)) for _ in range(n_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1))
                                   for _ in range(n_layers)])
        self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1))
                                   for _ in range(n_layers)])

        self.fc = nn.Linear(out_chs*seq_len, ans_size)

        # for param in self.parameters():
            # if len(param.size()) > 1:
                # nn.init.kaiming_normal(param)

        self.feat_size = out_chs

    def forward(self, x, feature=False):
        # x: (N, seq_len)

        # Embedding
        bs = x.size(0) # batch size
        # seq_len = x.size(1)
        # x = self.embedding(x) # (bs, seq_len, embd_size)

        # CNN
        x = x.unsqueeze(1) # (bs, Cin, seq_len, embd_size), insert Channnel-In dim
        # Conv2d
        #    Input : (bs, Cin,  Hin,  Win )
        #    Output: (bs, Cout, Hout, Wout)
        A = self.conv_0(x)      # (bs, Cout, seq_len, 1)
        A += self.b_0.repeat(1, 1, A.size(2), 1)
        B = self.conv_gate_0(x) # (bs, Cout, seq_len, 1)
        B += self.c_0.repeat(1, 1, B.size(2), 1)
        h = A * F.sigmoid(B)    # (bs, Cout, seq_len, 1)
        res_input = h # TODO this is h1 not h0

        for i, conv, conv_gate in zip(range(self.limit), self.conv, self.conv_gate):
            A = conv(h)
            A += self.b[i].repeat(1, 1, A.size(2), 1)
            B = conv_gate(h)
            B += self.c[i].repeat(1, 1, B.size(2), 1)
            h = A * F.sigmoid(B) # (bs, Cout, seq_len, 1)
            if i % self.res_block_count == 0: # size of each residual block
                h += res_input
                res_input = h

        if not feature:
            h = h.view(bs, -1) # (bs, Cout*seq_len)
            out = self.fc(h) # (bs, ans_size)
            out = F.log_softmax(out)
        else:
            h = torch.mean(h, 2) # mean over seq axis
            out = h.squeeze_()

        return out


_configs = {
    ConfigType.CNN_SMALL.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=16,
        conv1_size=(101, 8), conv1_pool=(1, 1), conv1_stride=(1, 1), dnn1_size=128, dnn2_size=128, tf_variant=False),
    ConfigType.CNN_FRAMES.value: dict(dropout_prob=0.5, height=20, width=40,  n_feature_maps1=128,
                                     conv1_size=(6, 8), conv1_pool=(2, 3), conv1_stride=(1, 1),
                                     conv2_size=(4, 4), conv2_pool=(1, 2), conv2_stride=(1, 1),
                                     n_feature_maps2=256,
                                     dnn1_size=512, tf_variant=True),
    # ConfigType.CNN_FRAMES.value: dict(dropout_prob=0.5, height=9, width=40,  n_feature_maps1=64,
    #                                   n_feature_maps2=64, conv1_size=(2, 3),
    #                                   conv1_pool=(1, 1), conv1_stride=(1, 1), conv2_size=(2, 3),
    #                                   conv2_stride=(1, 1), conv2_pool=(1, 1), tf_variant=True),
    ConfigType.CNN_TRAD_POOL2.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=64,
        n_feature_maps2=64, conv1_size=(20, 8), conv2_size=(10, 4), conv1_pool=(2, 2), conv1_stride=(1, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), tf_variant=True),
    ConfigType.CNN_ONE_STRIDE1.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=186,
        conv1_size=(101, 8), conv1_pool=(1, 1), conv1_stride=(1, 1), dnn1_size=128, dnn2_size=128, tf_variant=True),
    ConfigType.CNN_TSTRIDE2.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=78,
        n_feature_maps2=78, conv1_size=(16, 8), conv2_size=(9, 4), conv1_pool=(1, 3), conv1_stride=(2, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TSTRIDE4.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=100,
        n_feature_maps2=78, conv1_size=(16, 8), conv2_size=(5, 4), conv1_pool=(1, 3), conv1_stride=(4, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TSTRIDE8.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=126,
        n_feature_maps2=78, conv1_size=(16, 8), conv2_size=(5, 4), conv1_pool=(1, 3), conv1_stride=(8, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TPOOL2.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=94,
        n_feature_maps2=94, conv1_size=(21, 8), conv2_size=(6, 4), conv1_pool=(2, 3), conv1_stride=(1, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_TPOOL3.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=94,
        n_feature_maps2=94, conv1_size=(15, 8), conv2_size=(6, 4), conv1_pool=(3, 3), conv1_stride=(1, 1),
        conv2_stride=(1, 1), conv2_pool=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_ONE_FPOOL3.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=54,
        conv1_size=(101, 8), conv1_pool=(1, 3), conv1_stride=(1, 1), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_ONE_FSTRIDE4.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=186,
        conv1_size=(101, 8), conv1_pool=(1, 1), conv1_stride=(1, 4), dnn1_size=128, dnn2_size=128),
    ConfigType.CNN_ONE_FSTRIDE8.value: dict(dropout_prob=0.5, height=101, width=40,  n_feature_maps1=336,
        conv1_size=(101, 8), conv1_pool=(1, 1), conv1_stride=(1, 8), dnn1_size=128, dnn2_size=128),
    ConfigType.RES15.value: dict( use_dilation=True, n_layers=13, n_feature_maps=45),
    ConfigType.RES8.value: dict( n_layers=6, n_feature_maps=45, res_pool=(4, 3), use_dilation=False),
    ConfigType.RES26.value: dict( n_layers=24, n_feature_maps=45, res_pool=(2, 2), use_dilation=False),
    ConfigType.RES15_NARROW.value: dict( use_dilation=True, n_layers=13, n_feature_maps=19),
    ConfigType.RES15_WIDE.value: dict( use_dilation=True, n_layers=13, n_feature_maps=128),
    ConfigType.RES8_NARROW.value: dict( n_layers=6, n_feature_maps=19, res_pool=(4, 3), use_dilation=False),
    ConfigType.RES8_WIDE.value: dict( n_layers=6, n_feature_maps=128, res_pool=(4, 3), use_dilation=False),
    ConfigType.RES26_NARROW.value: dict( n_layers=24, n_feature_maps=19, res_pool=(2, 2), use_dilation=False),
    ConfigType.LSTM.value : dict( n_layers=3, h_dim=500)
}
