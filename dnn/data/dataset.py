import torch.utils.data as data
import librosa
import hashlib
import os
import random
import re
import numpy as np
from enum import Enum
from chainmap import ChainMap

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
        self.input_format = config["input_format"]

    @staticmethod
    def default_config():
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
        if silence:
            example = "__silence__"

        if random.random() < 0.7:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass

        in_len = self.input_length
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a:a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        use_clean = (self.set_type != DatasetType.TRAIN)
        if use_clean:
            bg_noise = np.zeros(in_len)

        if silence:
            data = np.zeros(in_len, dtype=np.float32)
        else:
            file_data = self._file_cache.get(example)
            data = librosa.core.load(example, sr=16000)[0] if file_data is None else file_data
            self._file_cache[example] = data

        if len(data) > in_len:
            # cliping the audio
            start_sample = np.random.randint(0, len(data) - in_len)
            data = data[start_sample:start_sample+in_len]
            # data = data[:in_len]
        else:
            # zero-padding the audio
            data = np.pad(data, (0, max(0, in_len - len(data))), "constant")

        if not use_clean:
            data = self._timeshift_audio(data)

        if random.random() < self.noise_prob or silence:
            a = random.random() * 0.1
            data = np.clip(a * bg_noise + data, -1, 1)

        audio_data = preprocess_audio(data, self.n_mels, self.filters, self.input_format)
        # data = torch.from_numpy(audio_data)
        self._audio_cache[example] = audio_data
        return audio_data

    @classmethod
    def read_manifest(cls, config):
        bg_folder = config["bkg_noise_folder"]
        sets = [{}, {}, {}]
        bg_noise_files = []

        try:
            train_files = open(config["train_manifest"], "r")
            val_files = open(config["val_manifest"], "r")
        except FileNotFoundError:
            train_files = None
            pass

        try:
            val_files = open(config["val_manifest"], "r")
        except FileNotFoundError:
            val_files = None
            pass

        try:
            test_files = open(config["test_manifest"], "r")
        except FileNotFoundError:
            test_files = None
            pass

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
    def read_df(cls, config, dataframes, label_set):
        bg_folder = config["bkg_noise_folder"]
        sets = [{}, {}, {}]
        bg_noise_files = []

        train_df, val_df, test_df = dataframes
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

        if test_df is not None:
            tag = DatasetType.TEST
            for idx, row in test_df.iterrows():
                path = os.path.join(data_dir, row.dir, row.file)
                label = label_set.index(row.spk)
                sets[tag.value][path] = label

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
        # train_pct = config["train_pct"]
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

class protoDataset(data.Dataset):

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
        self.window_stride = config.window_stride
        self.input_format = config.input_format

    @staticmethod
    def default_config():
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

        file_data = self._file_cache.get(example)
        data = librosa.core.load(example, sr=16000)[0] if file_data is None else file_data
        self._file_cache[example] = data

        in_len = self.input_length
        if len(data) > in_len:
            # chopping the audio
            start_sample = np.random.randint(0, len(data) - in_len)
            data = data[start_sample:start_sample+in_len]
            # data = data[:in_len]
        else:
            # zero-padding the audio
            data = np.pad(data, (0, max(0, in_len - len(data))), "constant")

        use_clean = self.set_type != DatasetType.TRAIN
        if not use_clean:
            data = self._timeshift_audio(data)

        data = preprocess_audio(data, self.n_mels, self.filters, self.input_format)
        # data = torch.from_numpy(audio_data).unsqueeze(0)
        self._audio_cache[example] = data
        return data

    @classmethod
    def read_train_manifest(cls, config):
        sets = [{}, {}]

        train_files = open(config.train_manifest, "r")
        val_files = open(config.val_manifest, "r")

        tag = DatasetType.TRAIN
        for sample in train_files:
            tokens = sample.rstrip().split(',')
            sets[tag.value][tokens[0]] = int(tokens[1])

        tag = DatasetType.DEV
        for sample in val_files:
            tokens = sample.rstrip().split(',')
            sets[tag.value][tokens[0]] = int(tokens[1])

        datasets = (cls(sets[0], DatasetType.TRAIN, config), cls(sets[1], DatasetType.DEV, config))
        return datasets

    @classmethod
    def read_val_manifest(cls, config):
        sets = [{}]

        val_files = open(config.val_manifest, "r")
        tag = DatasetType.DEV
        for sample in val_files:
            tokens = sample.rstrip().split(',')
            sets[tag.value][tokens[0]] = int(tokens[1])

        datasets = cls(sets[1], DatasetType.DEV, config)
        return datasets

    @classmethod
    def read_embed_manifest(cls, config):
        dataset = {}

        val_files = open(config.val_manifest, "r")
        for sample in val_files:
            tokens = sample.rstrip().split(',')
            dataset[tokens[0]] = tokens[1]

        dataset = cls(dataset, DatasetType.DEV, config)
        return dataset

    @classmethod
    def read_embed_df(cls, config, dataframe):
        dataset = {}

        data_dir = config.data_folder
        for idx, row in dataframe.iterrows():
            path = os.path.join(data_dir, row.file)
            dataset[path] = row.id

        dataset = cls(dataset, DatasetType.DEV, config)
        return dataset

    def __getitem__(self, index):
        return self.preprocess(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels)


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


