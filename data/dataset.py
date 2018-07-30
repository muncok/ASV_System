import librosa
import os
import random
import numpy as np
from enum import Enum

import torch
import torch.utils.data as data

from .manage_audio import preprocess_audio

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
        self.set_type = set_type
        # data
        self.audio_files = list(data.keys())
        self.audio_labels = list(data.values())
        # input augmentation
        self.noise_prob = config.get("noise_prob", 0)
        self.timeshift_ms = config["timeshift_ms"]
        # cache
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        # input audio config
        self.input_samples = config["input_samples"]
        self.input_format = config["input_format"]
        self.input_clip = config["input_clip"]
        # input feature config
        self.n_dct = config["n_dct_filters"]
        self.n_mels = config["n_mels"]
        self.filters = librosa.filters.dct(config["n_dct_filters"], config["n_mels"])
        self.window_size = config["window_size"]
        self.window_stride =  config["window_stride"]
        self.data_folder = config["data_folder"]

        if set_type == "train":
            # self._load_bg_noise(config["bkg_noise_folder"])
            self.bg_noise_audio = None
            self.random_clip = True
        else:
            self.bg_noise_audio = None
            self.random_clip = False

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

        in_len = self.input_samples
        # background noise
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a:a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        use_clean = (self.set_type != "train")
        if use_clean:
            bg_noise = np.zeros(in_len)

        file_data = self._file_cache.get(example)
        data = librosa.core.load(example, sr=16000)[0] if file_data is None else file_data
        self._file_cache[example] = data

        # input clipping
        if self.input_clip:
            if len(data) > in_len:
                if self.random_clip:
                    start_sample = np.random.randint(0, len(data) - in_len)
                else:
                    start_sample = 0
                data = data[start_sample:start_sample+in_len]
            else:
                data = np.pad(data, (0, max(0, in_len - len(data))), "constant")

        # time shift
        if not use_clean:
            data = self._timeshift_audio(data)

        # apply bg_noise to input
        if random.random() < self.noise_prob:
            a = random.random() * 0.1
            data = np.clip(a * bg_noise + data, -1, 1)

        # audio to input feature
        # mfcc hyper-parameters are hard coded.
        input_feature = preprocess_audio(data, self.n_mels, self.filters, self.input_format)
        self._audio_cache[example] = input_feature
        return input_feature.unsqueeze(0)

    @staticmethod
    def read_manifest(manifest_path):
        """
            It reads a manifest then returns sample dict
        :param config:
            config: contains options
        :return:
            sample dict
        """
        samples = {}
        manifest_file = open(manifest_path, "r")
        for sample in manifest_file:
            tokens = sample.rstrip().split(',')
            samples[tokens[0]] = int(tokens[1])
        return samples

    @classmethod
    def read_df(cls, config, df, set_type):
        if 'file' in df.columns:
            files = df.file.tolist()
        else:
            files = df.wav.tolist()

        if "label" in df.columns:
            labels = df.label.tolist()
        else:
            labels = [-1] * len(df)
        samples = dict(zip(files, labels))
        dataset = cls(samples, set_type, config)
        return dataset

    def _load_bg_noise(self, bg_folder):
        bg_noise_files = []
        for folder_name in os.listdir(bg_folder):
            path_name = os.path.join(bg_folder, folder_name)
            if os.path.isfile(path_name):
                continue
            elif folder_name == "_background_noise_":
                for filename in os.listdir(path_name):
                    wav_name = os.path.join(path_name, filename)
                    if os.path.isfile(wav_name):
                        bg_noise_files.append(wav_name)

        bg_noise_files = list(filter(lambda x: x.endswith("wav"),bg_noise_files))
        self.bg_noise_audio = [librosa.core.load(file, sr=16000)[0] for file in bg_noise_files]

    def __getitem__(self, index):
        return self.preprocess(os.path.join(self.data_folder, self.audio_files[index])), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels)

class featDataset(data.Dataset):
    def __init__(self, data, set_type, config):
        super().__init__()
        self.set_type = set_type
        # data
        self.files = list(data.keys())
        self.labels = list(data.values())
        self.data_folder = config["data_folder"]
        # cache
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        # input audio config
        self.input_frames = config["input_frames"]
        self.input_clip = config["input_clip"]
        self.input_dim = config["input_dim"]

        self.fail_count = 0

        if set_type == "train":
            self.random_clip = True
        else:
            self.random_clip = False

    def preprocess(self, example):

        file_data = self._file_cache.get(example)
        try:
            data = np.load(example) if file_data is None else file_data
        except FileNotFoundError:
            data = np.zeros((self.input_frames, self.input_dim))
            self.fail_count += 1
            print("{} is not found".format(example))
            # raise FileNotFoundError
        self._file_cache[example] = data

        # clipping
        in_len = self.input_frames
        if self.input_clip:
            if len(data) > in_len:
                if self.random_clip:
                    start_sample = np.random.randint(0, len(data) - in_len)
                else:
                    start_sample = 0
                data = data[start_sample:start_sample+in_len]
            else:
                data = np.pad(data, (0, max(0, in_len - len(data))), "constant")

        #TODO why do they have diffrent input dimension?
        data = data[:,:self.input_dim] # first dimension could be energy term
        data = torch.from_numpy(data).unsqueeze(0).float()
        return data

    @classmethod
    def read_df(cls, config, df, set_type):
        files = df.feat.tolist()

        if "label" in df.columns:
            labels = df.label.tolist()
        else:
            labels = [-1] * len(df)
        samples = dict(zip(files, labels))
        dataset = cls(samples, set_type, config)
        return dataset

    def __getitem__(self, index):
        return self.preprocess(os.path.join(self.data_folder, self.files[index])), self.labels[index]

    def __len__(self):
        return len(self.labels)
