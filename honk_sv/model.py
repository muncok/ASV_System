from enum import Enum
import hashlib
import math
import os
import random
import re

from chainmap import ChainMap
from torch.autograd import Variable
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from .manage_audio import preprocess_audio, fft_audio

class SimpleCache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.n_keys = 0

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value

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
    RES8_NARROW = "res8-narrow"
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

    def forward(self, x):
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

        return fc8_out

    def embedd(self, x):

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

    def forward(self, x, feature=False):
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
        if feature:
            return x
        else:
            return self.output(x)

class SpeechLSTMModel(SerializableModule):
    def __init__(self, config):
        super().__init__()
        target_size = config["n_labels"]
        no_cuda = config['no_cuda']
        self.input_dim = config["n_mels"]
        self.hidden_dim = config["h_dim"]
        self.n_layers = config["n_layers"]
        self.batch_size = config["batch_size"]

        self.lstm = nn.LSTM(input_size=self.input_dim , hidden_size=self.hidden_dim, num_layers=self.n_layers)
        self.proj = nn.Linear(self.hidden_dim, target_size)
        self.hidden = self.init_hidden(no_cuda=no_cuda)

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

    def forward(self, x):
        # input: (sequence, batch, features)
        lstm_out, hidden = self.lstm(x, self.hidden)
        last_out = lstm_out[-1]
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

    def forward(self, x, feature=False):
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
        if feature:
            return x
        else:
            return self.output(x)

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

# class frameModel(SpeechModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.input_frame = config['height']
#
#     def forward(self, x, feature=False):
#         valid_length = x.size(1) // self.input_frame
#         x = x.view()

class DatasetType(Enum):
    TRAIN = 0
    DEV = 1
    TEST = 2


class SpeechDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"
    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())
        config["bg_noise_files"] = list(filter(lambda x: x.endswith("wav"), config.get("bg_noise_files", [])))
        self.bg_noise_audio = [librosa.core.load(file, sr=16000)[0] for file in config["bg_noise_files"]]
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.n_dct = config["n_dct_filters"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self.filters = librosa.filters.dct(config["n_dct_filters"], config["n_mels"])
        self.n_mels = config["n_mels"]
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.window_size = config["window_size"]
        self.window_stride =  config["window_stride"]
        self.splice_len = config['splice_length']

    @staticmethod
    def default_config(dataset = None):
        config = {}
        config["silence_prob"] = 0.0
        config["noise_prob"] = 0.0
        config["n_dct_filters"] = 40
        config["input_length"] = 16000
        config["n_mels"] = 40
        config["timeshift_ms"] = 100
        config["unknown_prob"] = 0.0
        config["bkg_noise_folder"] = "/home/muncok/DL/dataset/SV_sets/speech_commands/_background_noise_"
        config["data_folder"] = "/home/muncok/DL/dataset/SV_sets"
        config["window_size"]= 0.025
        config["window_stride"]= 0.010
        return config

    def _timeshift_audio(self, data):
        shift = (16000 * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def preprocess(self, example, silence=False):
        # if silence:
        #     example = "__silence__"
        if random.random() < 0.7:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass
        # in_len = self.input_length
        # if self.bg_noise_audio:
        #     bg_noise = random.choice(self.bg_noise_audio)
        #     a = random.randint(0, len(bg_noise) - in_len - 1)
        #     bg_noise = bg_noise[a:a + in_len]
        # else:
        #     bg_noise = np.zeros(in_len)
        #
        # use_clean = (self.set_type != DatasetType.TRAIN)
        # if use_clean:
            # bg_noise = np.zeros(in_len)
        if silence:
            data = np.zeros(in_len, dtype=np.float32)
        else:
            file_data = self._file_cache.get(example)
            data = librosa.core.load(example, sr=16000)[0] if file_data is None else file_data
            self._file_cache[example] = data
        # if len(data) > in_len:
        #     start_frame = np.random.randint(0, len(data) - in_len)
        #     data = data[start_frame:start_frame+in_len]
        # else:
        #     data = np.pad(data, (0, max(0, in_len - len(data))), "constant")

        # if not use_clean:
        #     data = self._timeshift_audio(data)

        # if random.random() < self.noise_prob or silence:
        #     a = random.random() * 0.1
        #     data = np.clip(a * bg_noise + data, -1, 1)
        audio_data = preprocess_audio(data, self.n_mels, self.filters)
        if len(audio_data) < self.splice_len:
            audio_data = np.pad(audio_data, ((0, max(0, self.splice_len - len(audio_data))),(0,0)), "constant")
        data = torch.from_numpy(audio_data)
        # data = torch.from_numpy(fft_audio(data, self.window_size,self.window_stride))
        self._audio_cache[example] = data
        return data

    @classmethod
    def read_manifest(cls, config):
        bg_folder = config["bkg_noise_folder"]
        sets = [{}, {}, {}]
        bg_noise_files = []

        try:
            train_files = open(config["train_manifest"], "r") or None
            val_files = open(config["val_manifest"], "r") or None
        except FileNotFoundError:
            train_files = None
            val_files = None

        test_files = open(config["test_manifest"], "r") or None

        if train_files:
            tag = DatasetType.TRAIN
            train_samples = []
            for sample in train_files:
                tokens = sample.rstrip().split(',')
                train_samples.append(tokens)
            random.shuffle(train_samples)
            for tokens in train_samples:
                sets[tag.value][tokens[0]] = int(tokens[1]) if len(tokens)== 2 else [int(x) for x in tokens[1:]]

        if val_files:
            tag = DatasetType.DEV
            for sample in val_files:
                tokens = sample.rstrip().split(',')
                sets[tag.value][tokens[0]] = int(tokens[1]) if len(tokens)== 2 else [int(x) for x in tokens[1:]]

        if test_files:
            tag = DatasetType.TEST
            for sample in test_files:
                tokens = sample.rstrip().split(',')
                sets[tag.value][tokens[0]] = int(tokens[1]) if len(tokens)== 2 else [int(x) for x in tokens[1:]]

        for folder_name in os.listdir(bg_folder):
            path_name = os.path.join(bg_folder, folder_name)
            if os.path.isfile(path_name):
                continue
            elif folder_name == "_background_noise_":
                for filename in os.listdir(path_name):
                    wav_name = os.path.join(path_name, filename)
                    if os.path.isfile(wav_name):
                        bg_noise_files.append(wav_name)

        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(noise_prob=0), config)
        datasets = (cls(sets[0], DatasetType.TRAIN, train_cfg), cls(sets[1], DatasetType.DEV, test_cfg),
                cls(sets[2], DatasetType.TEST, test_cfg))
        return datasets

    @classmethod
    def command_splits(cls, config):
        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        words = {word: i + 2 for i, word in enumerate(wanted_words)}
        words.update({cls.LABEL_SILENCE:0, cls.LABEL_UNKNOWN:1})
        sets = [{}, {}, {}]
        unknowns = [0] * 3
        bg_noise_files = []
        unknown_files = []

        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name)
            is_bg_noise = False
            if os.path.isfile(path_name):
                continue
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_":
                is_bg_noise = True
            else:
                label = words[cls.LABEL_UNKNOWN]

            for filename in os.listdir(path_name):
                wav_name = os.path.join(path_name, filename)
                if is_bg_noise and os.path.isfile(wav_name):
                    bg_noise_files.append(wav_name)
                    continue
                elif label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    continue
                if config["group_speakers_by_id"]:
                    hashname = re.sub(r"_nohash_.*$", "", filename)
                max_no_wavs = 2**27 - 1
                bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                bucket = (bucket % (max_no_wavs + 1)) * (100. / max_no_wavs)
                if bucket < dev_pct:
                    tag = DatasetType.DEV
                elif bucket < test_pct + dev_pct:
                    tag = DatasetType.TEST
                else:
                    tag = DatasetType.TRAIN
                sets[tag.value][wav_name] = label

        for tag in range(len(sets)):
            unknowns[tag] = int(unknown_prob * len(sets[tag]))
        random.shuffle(unknown_files)
        a = 0
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unknown_files[a:b]}
            dataset.update(unk_dict)
            a = b

        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(noise_prob=0), config)
        datasets = (cls(sets[0], DatasetType.TRAIN, train_cfg), cls(sets[1], DatasetType.DEV, test_cfg),
                cls(sets[2], DatasetType.TEST, test_cfg))
        return datasets

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.preprocess(None, silence=True), 0
        return self.preprocess(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence

class MTLDataset(SpeechDataset):
    def __init__(self, data, set_type, config):
        super().__init__(data, set_type, config)
        self.labels= list(zip(*data.values()))
        self.spk_labels, self.sent_labels = list(zip(*data.values()))
        self.t_labels = len(self.audio_labels[0])
        self.n_silence = 0

    def __getitem__(self, index):
        if index >= len(self.spk_labels):
            return self.preprocess(None, silence=True), 0, 0
        return self.preprocess(self.audio_files[index]), self.spk_labels[index], self.sent_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence
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
    ConfigType.RES8_NARROW.value: dict( n_layers=6, n_feature_maps=19, res_pool=(4, 3), use_dilation=False),
    ConfigType.RES26_NARROW.value: dict( n_layers=24, n_feature_maps=19, res_pool=(2, 2), use_dilation=False),
    ConfigType.LSTM.value : dict( n_layers=3, h_dim=500)
}
