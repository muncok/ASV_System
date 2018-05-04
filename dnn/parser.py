# coding=utf-8
import numpy as np
from collections import ChainMap
import argparse

from .train import model as mod
from .data.dataset import SpeechDataset


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

def get_sv_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=100)

    parser.add_argument('-lr', '--learning_rate',
                        type=float,
                        help='learning rate for the model, default=0.001',
                        default=0.001)

    parser.add_argument('-lrS', '--lr_scheduler_step',
                        type=int,
                        help='StepLR learning rate scheduler step, default=20',
                        default=20)

    parser.add_argument('-lrG', '--lr_scheduler_gamma',
                        type=float,
                        help='StepLR learning rate scheduler gamma, default=0.5',
                        default=0.5)

    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)

    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=60)

    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=5)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=5)

    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=5)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=5)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=15)

    parser.add_argument('-nTestF', '--num_test_frames',
                        type=int,
                        help='number of samples per class to use as query for validation, default=15',
                        default=1)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda',
                        default=True)

    parser.add_argument('--input',
                        type=str,
                        help='model path to be loaded',
                        default=None)

    parser.add_argument('--output',
                        type=str,
                        help='model path to be saved',
                        default=None)

    parser.add_argument("--mode",
                        choices=["train", "eval", "sv_score", "posterior", "lda_train"],
                        default="train",
                        type=str)

    parser.add_argument('-inLen', '--input_length',
                        type=int,
                        help='length of input audio, sec',
                        default=3)

    parser.add_argument('-spLen', '--splice_length',
                        type=int,
                        help='length of spliced audio snippet, sec',
                        default=0.2)
    return parser

def test_config(model):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    config, _ = parser.parse_known_args()

    global_config = dict(model=model, no_cuda=False, n_epochs=500, lr=[0.001],
                         schedule=[np.inf], batch_size=64, dev_every=1, seed=0,
                         use_nesterov=False, input_file="", output_file="test.pt", gpu_no=0, cache_size=32768,
                         momentum=0.9, weight_decay=0.00001, num_workers = 16, print_step=1)
    builder = ConfigBuilder(
        mod.find_config(model),
        SpeechDataset.default_config(),
        global_config)

    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)
    mod_cls = mod.find_model(model)
    config["model_class"] = mod_cls
    return config
