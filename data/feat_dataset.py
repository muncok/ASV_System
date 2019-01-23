import os
import warnings
import numpy as np
from collections import OrderedDict

import torch
import torch.utils.data as data

def get_dir_path(file_path):
    return "/".join(file_path.split("/")[:-1])

class FeatDataset(data.Dataset):
    def __init__(self, data, set_type, config):
        super().__init__()
        self.set_type = set_type
        # data
        self.files = list(data.keys())
        self.labels = list(data.values())
        self.data_folder = config["data_folder"]
        # input audio config
        self.input_clip = config["input_clip"]
        if self.input_clip:
            self.input_frames = config["input_frames"]
        self.input_dim = config["input_dim"]
        if set_type == "train" and config['random_clip']:
            self.random_clip = True
        else:
            self.random_clip = False

    def preprocess(self, example):
        try:
            data = np.load(example)
        except FileNotFoundError:
                warnings.warn("{} is not found".format(example))
                raise FileNotFoundError
        # clipping
        if self.input_clip:
            in_len = self.input_frames
            if len(data) > in_len:
                if self.random_clip:
                    start_sample = np.random.randint(0, len(data) - in_len)
                else:
                    start_sample = 0
                data = data[start_sample:start_sample+in_len]
            else:
                gap = max(0, in_len - len(data))
                ## zero-padding
                # data = np.pad(data, (0, gap), "constant")
                # repeat, it shows better result than zero-padding
                repeat = int(np.floor(gap / len(data)))
                residual = gap % len(data)
                data = np.concatenate([np.tile(data, (repeat+1, 1)),
                    data[:residual]], axis=0)

        # first dimension represents energy
        # TODO: should include energy dimension?
        data = data[:,:self.input_dim]
        # expand a singleton dimension standing for a channel dimension
        data = torch.from_numpy(data).unsqueeze(0).float()
        return data

    @classmethod
    def read_df(cls, config, df, set_type):
        files = df.file.tolist()
        files = map(lambda x: x.rstrip(".npy") +".npy", files)

        if "label" in df.columns:
            labels = df.label.tolist()
        else:
            warnings.warn("No label data available")
            labels = [-1] * len(df)

        samples = OrderedDict(zip(files, labels))
        dataset = cls(samples, set_type, config)
        return dataset

    def __getitem__(self, index):
        return (self.preprocess(os.path.join(self.data_folder, self.files[index])),
                self.labels[index])

    def __len__(self):
        return len(self.labels)
