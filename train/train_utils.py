import os
import shutil
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from .angularLoss import AngleLoss

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

def load_checkpoint(config, model, optimizer):
    # https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/2
    input_file = config['input_file']
    if os.path.isfile(input_file):
        print("=> loading checkpoint '{}'".format(input_file))
        checkpoint = torch.load(input_file)
        config['s_epoch'] = checkpoint['epoch'] + 1
        config['best_metric'] = checkpoint['best_metric']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(input_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(input_file))

    return model, optimizer

def make_abspath(rel_path):
    if not os.path.isabs(rel_path):
        rel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel_path)
    return rel_path

def print_eval(name, scores, labels, loss, end="\n", verbose=False, binary=False):
    if isinstance(scores, tuple):
        scores = scores[0]
    batch_size = labels.size(0)
    if not binary:
        accuracy = (torch.max(scores, 1)[1] == labels.data).sum().float() / batch_size
    else:
        preds = (scores.data > 0.5)
        targets = (labels.data == 1)
        accuracy = (preds == targets).sum() / batch_size
    if verbose:
        tqdm.write("{} accuracy: {:>3}, loss: {:<7}".format(name, accuracy, loss), end=end)
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
    optimizer = torch.optim.SGD(model.parameters(),
            lr=config["lrs"][0], nesterov=config["use_nesterov"],
            weight_decay=config["weight_decay"],
            momentum=config["momentum"])

    return criterion, optimizer

