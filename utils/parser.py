# coding=utf-8
import numpy as np
from collections import ChainMap
import argparse

from ..data.dataset import SpeechDataset

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            # default_config's values are inserted through default argument
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def test_config(model):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    config, _ = parser.parse_known_args()

    global_config = dict(model=model, no_cuda=False, n_epochs=500, lr=[0.001],
                         schedule=[np.inf], batch_size=64, dev_every=1, seed=0,
                         use_nesterov=False, input_file="", output_file="test.pt", gpu_no=0, cache_size=32768,
                         momentum=0.9, weight_decay=0.00001, num_workers = 32, print_step=1)
    builder = ConfigBuilder(
        SpeechDataset.default_config(),
        global_config)

    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)
    return config

def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=140)

    parser.add_argument('-dataset',
                        type=str,
                        help='dataset',
                        default='voxc')

    parser.add_argument('-model',
                        type=str,
                        help='model',
                        choices=['TdnnModel', 'SimpleCNN', 'TdnnStatModel']
                        )

    parser.add_argument('-suffix',
                        type=str,
                        help='suffix for model.pt name',
                        default='')

    parser.add_argument('-inSec', '--input_seconds',
                        type=float,
                        help='length of input audio, sec',
                        default=3)

    parser.add_argument('-spSec', '--splice_seconds',
                        type=float,
                        help='length of spliced audio snippet, sec',
                        default=0.2)

    parser.add_argument('-stSec', '--stride_seconds',
                        type=float,
                        help='interval of audio snippets, sec',
                        default=1)

    parser.add_argument('-s_epoch', '--start_epoch',
                        type=int,
                        help='where the epoch starts',
                        default=0)
    parser.add_argument('-cuda',
                        action = 'store_true',
                        default= False)
    return parser
