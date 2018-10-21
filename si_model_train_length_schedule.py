# coding: utf-8
import os
import sys
# import uuid

import torch
from tensorboardX import SummaryWriter

from utils.parser import train_parser, set_train_config

from data.dataloader import init_loaders
from data.data_utils import find_dataset, find_trial

from model.model_utils import find_model

from train.train_utils import (set_seed, find_optimizer,
        load_checkpoint, save_checkpoint, new_exp_dir)
from train.train_utils import Logger
from train.si_train import sub_utter_val, batch_variable_length_train
from eval.sv_test import sv_test
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.optim.lr_scheduler import MultiStepLR



#########################################
# Parser
#########################################
parser = train_parser()
args = parser.parse_args()
dataset = args.dataset
config = set_train_config(args)


#########################################
# Dataset loaders
#########################################
_, datasets = find_dataset(config)
loaders = init_loaders(config, datasets)

#########################################
# Model Initialization
#########################################
model= find_model(config)
criterion, optimizer = find_optimizer(config, model)
plateau_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10)

if config['input_file']:
    # start new experiment continuing from "input_file"
    load_checkpoint(config, model, criterion, optimizer)
#########################################
# Model Save Path
#########################################

if config['output_dir']:
    pass
else:
    # start new experiment
    config['output_dir'] = new_exp_dir(config)

if not os.path.isdir(config['output_dir']):
    os.makedirs(config['output_dir'])

print("Model will be saved to : {}".format(config['output_dir']))

#########################################
# Logger
#########################################
# tensorboard
log_dir = os.path.join(config['output_dir'], 'logs')
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir)
sys.stdout = Logger(os.path.join(config['output_dir'],
    'logs/{}'.format('train_log.txt')))

print(' '.join(sys.argv))
# sys.stdout = Logger(os.path.join(config['output_dir'],
    # 'logs/log_{}'.format(str(uuid.uuid4())[:5]) + '.txt'))


#########################################
# dataloader and scheduler
#########################################
if not config['no_eer']:
    train_loader, val_loader, test_loader, sv_loader = loaders
else:
    train_loader, val_loader, test_loader = loaders


#########################################
# trial
#########################################
trial = find_trial(config)
if not config['no_eer']:
    best_metric = config['best_metric'] if 'best_metric' in config \
            else 1.0
else:
    best_metric = config['best_metric'] if 'best_metric' in config \
            else 0.0

#########################################
# Model Training
#########################################

set_seed(config)
for epoch_idx in range(config["s_epoch"], config["n_epochs"]):
    config['epoch_idx'] = epoch_idx
    curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
    # idx = 0
    # while(epoch_idx >= config['lr_schedule'][idx]):
    # # use new lr from schedule epoch not a next epoch
        # idx += 1
        # if idx == len(config['lr_schedule']):
            # break
    # curr_lr = config['lrs'][idx]
    # optimizer.state_dict()['param_groups'][0]['lr'] = curr_lr
    print("curr_lr: {}".format(curr_lr))

    # train code
    train_loss, train_acc = batch_variable_length_train(config, train_loader, model, optimizer, criterion)
    writer.add_scalar("train/lr", curr_lr, epoch_idx)
    writer.add_scalar('train/loss', train_loss, epoch_idx)
    writer.add_scalar('train/acc', train_acc, epoch_idx)

    # validation code
    val_loss, val_acc = sub_utter_val(config, val_loader, model, criterion)
    writer.add_scalar('val/loss', val_loss, epoch_idx)
    writer.add_scalar('val/acc', val_acc, epoch_idx)
    print("epoch #{}, val accuracy: {}".format(epoch_idx, val_acc))

    plateau_scheduler.step(train_loss)

    # evaluate best_metric
    if not config['no_eer']:
        # eer validation code
        eer, label, score = sv_test(config, sv_loader, model, trial)
        current_metric = eer
        writer.add_scalar('sv_eer', eer, epoch_idx)
        writer.add_pr_curve('DET', label, score, epoch_idx)
        print("epoch #{}, sv eer: {}".format(epoch_idx, eer))
        if eer < best_metric:
            best_metric = eer
            is_best = True
        else:
            is_best = False
    else:
        current_metric = val_acc
        if val_acc > best_metric:
            best_metric = val_acc
            is_best = True
        else:
            is_best = False

    # for name, param in model.named_parameters():
        # writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx)

    filename = config["output_dir"] + \
            "/model.{:.4}.pth.tar".format(curr_lr)

    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    save_checkpoint({
        'epoch': epoch_idx,
        'step_no': (epoch_idx+1) * len(train_loader),
        'arch': config["arch"],
        'n_labels': config["n_labels"],
        'dataset': config["dataset"],
        'loss': config["loss"],
        'state_dict': model_state_dict,
        'best_metric': current_metric,
        'optimizer' : optimizer.state_dict(),
        }, epoch_idx, is_best, filename=filename)

#########################################
# Model Evaluation
#########################################
test_loss, test_acc = sub_utter_val(config, test_loader, model, criterion)
