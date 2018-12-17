import os
import shutil
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

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

        if 'step_no' in checkpoint:
            config['step_no'] = checkpoint['step_no']

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
        raise FileNotFoundError


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
    criterion = nn.CrossEntropyLoss()

    return criterion

def find_optimizer(config, model):
    criterion = nn.CrossEntropyLoss()

    # optimizer
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
        ("saved_models/{dset}/{suffix}/{arch}/{in_format}{in_dim}_{s_len1}f_{s_len2}f").format(
                dset=config['dataset'], arch=config['arch'],
                s_len1=config["splice_frames"][0],
                s_len2=config["splice_frames"][-1],
                suffix=config["suffix"],
                in_format=config['input_format'],
                in_dim=config["input_dim"])

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

def train(config, train_loader, model, optimizer, criterion):
    model.train()
    loss_sum = 0
    accs = []

    print_steps = (np.arange(1,21) * 0.1
                    * len(train_loader)).astype(np.int64)

    splice_frames = config['splice_frames']
    if len(splice_frames) > 1:
        splice_frames_ = np.random.randint(splice_frames[0], splice_frames[1])
    else:
        splice_frames_ = splice_frames[-1]

    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.narrow(2, 0, splice_frames_)

        if not config["no_cuda"]:
            X = X.cuda()
            y = y.cuda()

        optimizer.zero_grad()

        scores = model(X)

        loss = criterion(scores, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        # schedule over iteration
        accs.append(print_eval("train step #{}".format('0'), scores, y,
            loss_sum/(batch_idx+1), display=False))

        del scores
        del loss
        if batch_idx in print_steps:
            print("[{}/{}] train, loss: {:.4f}, acc: {:.5f} ".format(
                batch_idx, len(train_loader),
                loss_sum/(batch_idx+1), np.mean(accs)))

    avg_acc = np.mean(accs)

    return loss_sum, avg_acc

def val(config, val_loader, model, criterion):
    with torch.no_grad():
        model.eval()
        accs = []
        loss_sum = 0
        for (X, y) in val_loader:
            if not config["no_cuda"]:
                X = X.cuda()
                y = y.cuda()
            scores = model(X)
            loss = criterion(scores, y)
            loss_sum += loss.item()
            accs.append(print_eval("dev", scores, y,
                loss.item()))
        avg_acc = np.mean(accs)

        return loss_sum, avg_acc
