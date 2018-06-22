from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .model import SerializableModule, truncated_normal
from .Angular_loss import AngleLinear

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
    CNN_LONG = "cnn-long"
    CNN_FRAMES = "cnn-frames"

def find_config(conf):
    '''
    It find config which will bed used to init a model.
    :param conf:
    :return: model config
    '''
    if isinstance(conf, ConfigType):
        conf = conf.value
    if conf in _configs:
        return _configs[conf]
    else:
        return {}

class SpeechModel(SerializableModule):
    def __init__(self, config, model, n_labels):
        super().__init__()
        # n_labels = config["n_labels"]
        config = find_config(model)
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

        with torch.no_grad():
            x = torch.zeros(1, 1, height, width)
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

class SpeechResModel(SerializableModule):
    def __init__(self, arch, n_labels):
        super().__init__()
        # n_labels = config["n_labels"]
        config = find_config(arch)
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
        self.output = nn.Linear(n_maps, n_maps)
        self.anglelinear = AngleLinear(n_maps, n_labels)
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
        x = self.output(x)
        x = self.anglelinear(x)
        return x


_configs = {
    ConfigType.CNN_FRAMES.value: dict(dropout_prob=0.5, height=20, width=40,  n_feature_maps1=128,
                                     conv1_size=(6, 8), conv1_pool=(2, 3), conv1_stride=(1, 1),
                                     conv2_size=(4, 4), conv2_pool=(1, 2), conv2_stride=(1, 1),
                                     n_feature_maps2=256,
                                     dnn1_size=512, tf_variant=True),
    ConfigType.CNN_LONG.value: dict(dropout_prob=0.5, height=301, width=40,  n_feature_maps1=64,
                                          n_feature_maps2=64, conv1_size=(20, 8), conv2_size=(10, 4), conv1_pool=(2, 2), conv1_stride=(1, 1),
                                          conv2_stride=(1, 1), conv2_pool=(1, 1), tf_variant=True),
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
