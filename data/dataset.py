import os
import random
import librosa
import numpy as np
from collections import OrderedDict

import torch.utils.data as data
from .manage_audio import preprocess_audio

def get_dir_path(file_path):
    return "/".join(file_path.split("/")[:-1])

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
        # input audio config
        self.input_samples = config["input_samples"]
        self.input_format = config["input_format"]
        self.input_clip = config["input_clip"]
        # input feature config
        self.n_dct = config["n_dct_filters"]
        self.n_mels = config["n_mels"]
        self.filters = librosa.filters.dct(config["n_dct_filters"], config["n_mels"])
        self.data_folder = config["data_folder"]

        if set_type == "train":
            bg_folder = os.path.join(config['data_folder'],"_background_noise_")
            self.apply_bg_noise = False
            if os.path.isdir(bg_folder):
                self._load_bg_noise(bg_folder)
                self.apply_bg_noise = True
            self.random_clip = True
        else:
            self.apply_bg_noise = None
            self.random_clip = False

    def _timeshift_audio(self, data):
        shift = (16000 * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def preprocess(self, example):
        in_len = self.input_samples

        # background noise
        if self.apply_bg_noise:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a:a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        use_clean = (self.set_type != "train")
        if use_clean:
            bg_noise = np.zeros(in_len)

        # ======= librosa ======
        data = librosa.core.load(example, sr=16000)[0]

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

        #_name apply bg_noise to input
        if random.random() < self.noise_prob:
            a = random.random() * 0.1
            data = np.clip(a * bg_noise + data, -1, 1)

        # audio to input feature
        # mfcc hyper-parameters are hard coded.
        input_feature = preprocess_audio(data, self.n_mels, self.filters, self.input_format)
        return input_feature.unsqueeze(0)

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
        samples = OrderedDict(zip(files, labels))
        dataset = cls(samples, set_type, config)
        return dataset

    def _load_bg_noise(self, bg_folder):
        bg_noise_files = []
        for filename in os.listdir(bg_folder):
            wav_name = os.path.join(bg_folder, filename)
            if os.path.isfile(wav_name):
                bg_noise_files.append(wav_name)

        bg_noise_files = list(filter(lambda x: x.endswith("wav"), bg_noise_files))
        self.bg_noise_audio = [librosa.core.load(file, sr=16000)[0] for file in bg_noise_files]

    def __getitem__(self, index):
        return self.preprocess(os.path.join(self.data_folder, self.audio_files[index])), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels)

