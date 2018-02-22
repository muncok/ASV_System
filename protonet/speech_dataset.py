import os
import librosa
import random
import numpy as np
from enum import Enum
from chainmap import ChainMap
import torch
import torch.utils.data as data

from manage_audio import preprocess_audio

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


class DatasetType(Enum):
    TRAIN = 0
    DEV = 1
    TEST = 2


class SpeechDataset(data.Dataset):
    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())
        self.n_dct = config.n_dct_filters
        self.input_length = config.input_length
        self.timeshift_ms = config.timeshift_ms
        self.filters = librosa.filters.dct(config.n_dct_filters, config.n_mels)
        self.n_mels = config.n_mels
        self._audio_cache = SimpleCache(config.cache_size)
        self._file_cache = SimpleCache(config.cache_size)
        self.window_size = config.window_size
        self.window_stride =  config.window_stride

    @staticmethod
    def default_config(dataset = None):
        config = {}
        config.n_dct_filters = 40
        config.input_length = 16000
        config.n_mels = 40
        config.timeshift_ms = 100
        config.data_folder = "/home/muncok/DL/dataset/SV_sets"
        config.window_size= 0.025
        config.window_stride= 0.010
        return config

    def _timeshift_audio(self, data):
        shift = (16000 * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def preprocess(self, example):
        if random.random() < 0.7:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass

        in_len = self.input_length

        file_data = self._file_cache.get(example)
        data = librosa.core.load(example, sr=16000)[0] if file_data is None else file_data
        self._file_cache[example] = data

        if len(data) > in_len:
            # cliping the audio
            start_sample = np.random.randint(0, len(data) - in_len)
            data = data[start_sample:start_sample+in_len]
        else:
            # zero-padding the audio
            data = np.pad(data, (0, max(0, in_len - len(data))), "constant")

        use_clean = self.set_type == DatasetType.TRAIN
        if not use_clean:
            data = self._timeshift_audio(data)

        audio_data = preprocess_audio(data, self.n_mels, self.filters)
        # if len(audio_data) < self.splice_len:
            # audio_data = np.pad(audio_data, ((0, max(0, self.splice_len - len(audio_data))),(0,0)), "constant")
        data = torch.from_numpy(audio_data).unsqueeze(0)
        self._audio_cache[example] = data
        return data

    @classmethod
    def read_manifest(cls, config):
        sets = [{},{}]

        train_files = open(config.train_manifest, "r")
        val_files = open(config.val_manifest, "r")
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

        datasets = (cls(sets[0], DatasetType.TRAIN, config), cls(sets[1], DatasetType.DEV, config))

        return datasets

    @classmethod
    def read_df(cls, config, dataframes, label_set):
        sets = [{}, {}, {}]

        train_df, val_df= dataframes
        data_dir = config["data_folder"]

        if train_df is not None:
            tag = DatasetType.TRAIN
            for idx, row in train_df.iterrows():
                path = os.path.join(data_dir, row.dir, row.file)
                label = label_set.index(row.spk)
                sets[tag.value][path] = label

        if val_df is not None:
            tag = DatasetType.DEV
            for idx, row in val_df.iterrows():
                path = os.path.join(data_dir, row.dir, row.file)
                label = label_set.index(row.spk)
                sets[tag.value][path] = label


        train_cfg = config
        test_cfg = config
        datasets = (cls(sets[0], DatasetType.TRAIN, train_cfg), cls(sets[1], DatasetType.DEV, test_cfg))
        return datasets

    def __getitem__(self, index):
        return self.preprocess(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels)

