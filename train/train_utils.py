import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from .angularLoss import AngleLoss


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

def find_criterion(config, model_type, n_labels):
    if config["loss"] == "softmax":
        criterion = nn.CrossEntropyLoss()
    elif config["loss"] == "angular":
        criterion = AngleLoss()
    else:
        raise NotImplementedError
    return criterion

def save_before_lr_change(config, model, new_lr):
    print("saving model before the lr changed")
    # save the model before the lr change
    model.save(config["output_file"].rstrip('.pt')+".{:.4}.pt".format(new_lr))

def get_dir_path(file_path):
    return "/".join(file_path.split("/")[:-1])
