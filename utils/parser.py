# coding=utf-8
from collections import ChainMap
import argparse

def secToSample(sec):
    return int(16000 * sec)

def secToFrames(sec):
    return secToSample(sec)//160+1

def framesToSample(fr):
    return (fr-1)*160

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

def default_audio_config():
    config = {}
    config["noise_prob"] = 0.0
    config["n_dct_filters"] = 40
    config["input_samples"] = 16000
    config["n_mels"] = 40
    config["timeshift_ms"] = 100
    config["bkg_noise_folder"] = "/home/muncok/DL/dataset/SV_sets/speech_commands/_background_noise_"
    config["window_size"]= 0.025
    config["window_stride"]= 0.010
    return config

def default_config():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    config, _ = parser.parse_known_args()

    global_config = dict(
            dev_every=1,
            use_nesterov=False,
            gpu_no=[0], cache_size=32768,
            momentum=0.9, weight_decay=0.0001, num_workers=16,
            print_step=100)

    builder = ConfigBuilder(
        default_audio_config(),
        global_config)

    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)
    return config

def set_train_config(args):
    #TODO test
    config = default_config()
    config_from_agrs = vars(args)
    config.update(config_from_agrs)

    config['input_clip'] = True
    config['input_samples'] = framesToSample(args.input_frames)
    config['no_cuda'] = not args.cuda
    config['score_mode'] = 'approx'

    assert len(config['lrs']) == len(config['lr_schedule'])+1,\
            "invalid lr scheduling"

    return config

def set_score_config(args):
    config = default_config()
    config_from_agrs = vars(args)
    config.update(config_from_agrs)

    if args.score_mode == "precise":
        config['input_clip'] = False
        config['batch_size'] = 1
    else:
        config['input_clip'] = True
        config['batch_size'] = args.batch_size
    config['no_cuda'] = not args.cuda

    return config

def train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nep', '--n_epochs',
                        type=int,
                        help='number of epochs to train for',
                        default=140)

    parser.add_argument('-batch', '--batch_size',
                        type=int,
                        help='batch size',
                        default=64)

    parser.add_argument('-lrs', '--lrs',
                        type=float,
                        nargs='+',
                        help='learning lates')

    parser.add_argument('-sche', '--lr_schedule',
                        type=int,
                        nargs='+',
                        help='check points of changing learning lates',
                        default=[])

    parser.add_argument('-dataset',
                        type=str,
                        help='dataset',
                        default='')

    parser.add_argument('-arch',
                        type=str,
                        help='type of model',
                        )

    parser.add_argument('-loss',
                        type=str,
                        help='type of loss',
                        choices=['softmax', 'angular'],
                        default='softmax'
                        )

    # parser.add_argument('-version',
                        # type=float,
                        # help='version of si_train code',
                        # choices=[0, 1, 2, 2.1, 3],
                        # default=1
                        # )

    parser.add_argument('-input_file',
                        type=str,
                        help='model path to be loaded',
                        default=None,
                        )

    parser.add_argument('-output_file',
                        type=str,
                        help='model path to be saved',
                        default="test.pt",
                        )

    parser.add_argument('-s_epoch',
                        type=int,
                        help='where the epoch starts',
                        default=0)

    parser.add_argument('-suffix',
                        type=str,
                        help='suffix for model.pt name',
                        default='')

    parser.add_argument('-inFm', '--input_format',
                        type=str,
                        help='input feature, mfcc, fbank',
                        default="fbank")

    parser.add_argument('-inFr', '--input_frames',
                        type=int,
                        help='number of input frames, frames',
                        default=201)

    parser.add_argument('-spFr', '--splice_frames',
                        type=int,
                        nargs='+',
                        help='number of spliced frames, frames',
                        default=[21])

    parser.add_argument('-stFr', '--stride_frames',
                        type=int,
                        help='moving stride interval, frames',
                        default=1)

    parser.add_argument('-seed',
                        type=int,
                        help='training seed',
                        default=1337
                        )

    parser.add_argument('-gpu_no',
                        type=int,
                        nargs='+',
                        help='gpu device ids',
                        default=[0]
                        )

    parser.add_argument('-cuda',
                        action = 'store_true',
                        default= False)
    return parser

def score_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch', '--batch_size',
                        type=int,
                        help='batch size',
                        default=64)

    parser.add_argument('-dataset',
                        type=str,
                        help='dataset',
                        default='')

    parser.add_argument('-arch',
                        type=str,
                        help='type of model',
                        )

    parser.add_argument('-loss',
                        type=str,
                        help='type of loss',
                        choices=['softmax', 'angular'],
                        default='softmax'
                        )

    parser.add_argument('-input_file',
                        type=str,
                        help='model path to be loaded',
                        default=None,
                        )

    parser.add_argument('-inFm', '--input_format',
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

    parser.add_argument('-cuda',
                        action = 'store_true',
                        default= False)

    parser.add_argument('-score_mode',
                        type=str,
                        help='precision of scoring',
                        choices=['precise', 'approx'],
                        default='approx'
                        )

    parser.add_argument('-output_folder',
                        type=str,
                        help='path to be saved',
                        )

    parser.add_argument('-gpu_no',
                        type=int,
                        nargs='+',
                        help='gpu device ids',
                        default=[0]
                        )

    return parser

