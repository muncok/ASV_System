import sys
import os
import errno
import shutil
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from ..model.model_utils import find_model
from .angularLoss import AngleLoss

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def get_dir_path(file_path):
    return "/".join(file_path.split("/")[:-1])

def save_checkpoint(state, epoch_idx, is_best, filename='checkpoint.pth.tar'):
    # https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/2

    print("=> saving checkpoint")
    torch.save(state, filename)
    print("=> saved checkpoint '{} (epoch {})'".format(filename, epoch_idx))
    if is_best:
        print("best score!!")
        shutil.copyfile(filename, os.path.join(get_dir_path(filename),
            'model_best.pth.tar'))

def load_checkpoint(config, model=None, criterion=None, optimizer=None):
    # https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/2
    input_file = config['input_file']
    if os.path.isfile(input_file):
        print("=> loading checkpoint '{}'".format(input_file))
        checkpoint = torch.load(input_file)
        config['s_epoch'] = checkpoint['epoch'] + 1
        config['best_metric'] = checkpoint['best_metric']
        config['arch'] = checkpoint['arch']

        if 'loss' in checkpoint:
            config['loss'] = checkpoint['loss']

        if 'step_no' in checkpoint:
            config['step_no'] = checkpoint['step_no']
            if config['loss'] == 'angular' and criterion is not None:
                # for lambda annealling
                criterion.it = config['step_no']
                print("start iteration {}".format(criterion.it))

        if not model:
            # model was not loaded yet.
            try:
                if 'softmax' in input_file or checkpoint['loss'] == 'softmax':
                    n_labels = checkpoint['state_dict']['output.weight'].shape[0]
                else:
                    n_labels = checkpoint['state_dict']['output.weight'].shape[1]
            except:
                n_labels = config['n_labels']
            config['n_labels'] = n_labels
            model = find_model(config)

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])

        if optimizer:
            opt_state_dict = checkpoint['optimizer']
            opt_state_dict['param_groups'][0]['lr'] = config['lrs'][0]
            optimizer.load_state_dict(opt_state_dict)
        print("=> loaded checkpoint '{}' (epoch {}), score: {:.5f}"
              .format(input_file, checkpoint['epoch'],
                  checkpoint['best_metric']))
    else:
        print("=> no checkpoint found at '{}'".format(input_file))

    return model, optimizer

def make_abspath(rel_path):
    if not os.path.isabs(rel_path):
        rel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel_path)
    return rel_path

def print_eval(name, scores, labels, loss, end="\n", display=False, binary=False):
    if isinstance(scores, tuple):
        scores = scores[0]
    batch_size = labels.size(0)
    if not binary:
        accuracy = (torch.max(scores, 1)[1] == labels.data).sum().float() / batch_size
    else:
        preds = (scores.data > 0.5)
        targets = (labels.data == 1)
        accuracy = (preds == targets).sum() / batch_size
    if display:
        tqdm.write("{} accuracy: {:.3f}, loss: {:.7f}".format(name, accuracy, loss), end=end)
    return accuracy

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    # torch.cuda.cudnn_enabled = False
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def find_criterion(config, model):
    if config["loss"] == "softmax":
        criterion = nn.CrossEntropyLoss()
    elif config["loss"] == "angular":
        criterion = AngleLoss()
    else:
        raise NotImplementedError

    return criterion

def find_optimizer(config, model):
    if config["loss"] == "softmax":
        criterion = nn.CrossEntropyLoss()
    elif config["loss"] == "angular":
        criterion = AngleLoss()
    else:
        raise NotImplementedError

    # optimizer
    # learnable_params = [param for param in model.parameters() \
    # if param.requires_grad == True]
    init_lr = config["lrs"][0]
    optimizer = torch.optim.SGD(model.parameters(),
            lr=init_lr, nesterov=config["use_nesterov"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"])

    return criterion, optimizer

def new_exp_dir(config, old_exp_dir=None):
    # suffix: v1, v2 ...
    if not old_exp_dir:
        old_exp_dir = \
        ("models/{dset}/{suffix}/{arch}_{loss}/{in_format}_{s_len1}f_{s_len2}f").format(
                dset=config['dataset'], arch=config['arch'],
                loss=config["loss"],
                s_len1=config["splice_frames"][0],
                s_len2=config["splice_frames"][-1],
                in_format=config["input_format"],
                suffix=config["suffix"])

    done = False
    v = 0
    while not done:
        output_dir_ = "{output_dir}_v{version:02d}".format(
                output_dir=old_exp_dir, version=v)
        # now we check model_best.pth.tar directly
        if not os.path.isfile(output_dir_+"/model_best.pth.tar"):
            output_dir = output_dir_
            if not os.path.isdir(output_dir_):
                os.makedirs(output_dir)
            done = True
        else:
            v += 1

    return output_dir
