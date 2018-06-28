# coding=utf-8
from collections import ChainMap
import argparse

from ..data.dataset import SpeechDataset
from . import framesToSample

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

def default_config(model):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    config, _ = parser.parse_known_args()

    global_config = dict(model=model, n_epochs=80, dev_every=1,
            seed=0, use_nesterov=False, input_file="",
            output_file="test.pt", gpu_no=0, cache_size=32768,
            momentum=0.9, weight_decay=0.0001, num_workers = 16,
            print_step=1)

    builder = ConfigBuilder(
        SpeechDataset.default_config(),
        global_config)

    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)
    return config

def set_input_config(config, args):
    config['input_clip'] = True
    config['input_frames'] = args.input_frames
    config['input_samples'] = framesToSample(args.input_frames)
    config['splice_frames'] = args.splice_frames
    config['stride_frames'] = args.stride_frames
    config['input_format'] = args.input_format
    return config

def set_train_config(config, args):
    config['input_file'] = args.input_file
    config['n_epochs'] = args.epochs
    config['s_epoch'] = args.start_epoch
    config['lr'] = args.lrs
    config['schedule'] = args.lr_schedule
    config['batch_size'] = args.batch_size
    config['no_cuda'] = not args.cuda
    config['suffix'] = args.suffix
    return config

def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nep', '--epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=140)

    parser.add_argument('-lrs', '--lrs',
                        type=float,
                        nargs='+',
                        help='learning lates',
                        default=[0.01, 0.001])

    parser.add_argument('-lr_sch', '--lr_schedule',
                        type=int,
                        nargs='+',
                        help='check points of changing learning lates',
                        default=[1e4, 5e4])

    parser.add_argument('-batch', '--batch_size',
                        type=int,
                        help='batch size',
                        default=64)

    parser.add_argument('-dataset',
                        type=str,
                        help='dataset',
                        default='')

    parser.add_argument('-model',
                        type=str,
                        help='type of model',
                        )

    parser.add_argument('-input_file',
                        type=str,
                        help='model path to be loaded',
                        default=None,
                        )

    parser.add_argument('-suffix',
                        type=str,
                        help='suffix for model.pt name',
                        default='')

    parser.add_argument('-inFo', '--input_format',
                        type=str,
                        help='input feature, mfcc, fbank',
                        default="fbank")

    parser.add_argument('-inFr', '--input_frames',
                        type=int,
                        help='number of input frames, frames',
                        default=201)

    parser.add_argument('-spFr', '--splice_frames',
                        type=int,
                        help='number of spliced frames, frames',
                        default=21)

    parser.add_argument('-stFr', '--stride_frames',
                        type=int,
                        help='moving stride interval, frames',
                        default=1)

    # parser.add_argument('-inDim', '--input_dim',
                        # type=int,
                        # help='input_dimension',
                        # default=40)

    parser.add_argument('-s_epoch', '--start_epoch',
                        type=int,
                        help='where the epoch starts',
                        default=0)

    parser.add_argument('-cuda',
                        action = 'store_true',
                        default= False)
    return parser

