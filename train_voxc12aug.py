# coding: utf-8
import os
import sys
import pandas as pd
# import uuid

import torch
from tensorboardX import SummaryWriter

from utils.parser import train_parser, set_train_config
from utils.utils import Logger

from data.dataloader import init_loaders
from data.feat_dataset import FeatDataset
from model.model_utils import find_model

from system.si_train import set_seed, find_optimizer
from system.si_train import load_checkpoint, save_checkpoint, new_exp_dir
from system.si_train import train, val
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
name, in_format, in_dim, mode = dataset.split("_")
config['data_folder'] = "/dataset/SV_sets/voxceleb12/feats/fbank64_vad"
config['input_format'] = in_format
config['input_dim'] = int(in_dim)
config['num_workers'] = 8
config['n_labels'] = 7365
df = pd.read_csv("/dataset/SV_sets/voxceleb12/dataframes/voxc12aug_si.csv")
train_df = df[df.set == 'train']
val_df = df[df.set == 'val']
train_dataset = FeatDataset.read_df(config, train_df, 'train')
val_dataset = FeatDataset.read_df(config, val_df, 'val')
loaders = init_loaders(config, [train_dataset, val_dataset])

#########################################
# Model Initialization
#########################################
model = find_model(config)
criterion, optimizer = find_optimizer(config, model)
plateau_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

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

print("=> Model will be saved to: {}".format(config['output_dir']))

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
    train_loader, val_loader, sv_loader = loaders
else:
    train_loader, val_loader = loaders

#########################################
# Model Training
#########################################
set_seed(config)
best_metric = 0.0
for epoch_idx in range(config["s_epoch"], config["n_epochs"]):
    config['epoch_idx'] = epoch_idx
    curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
    print("===============================================")
    print("epoch #{}".format(epoch_idx))
    print("curr_lr: {}".format(curr_lr))

    # train code
    train_loss, train_acc = train(config, train_loader, model, optimizer, criterion)
    writer.add_scalar("train/lr", curr_lr, epoch_idx)
    writer.add_scalar('train/loss', train_loss, epoch_idx)
    writer.add_scalar('train/acc', train_acc, epoch_idx)
    plateau_scheduler.step(train_loss)

    # validation code
    val_loss, val_acc = val(config, val_loader, model, criterion)
    writer.add_scalar('val/loss', val_loss, epoch_idx)
    writer.add_scalar('val/acc', val_acc, epoch_idx)
    print("val acc: {}".format(val_acc))

    current_metric = val_acc
    if val_acc > best_metric:
        best_metric = val_acc
        is_best = True
    else:
        is_best = False

    # for name, param in model.named_parameters():
        # writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx)

    filename = config["output_dir"] + "/model.{:.4}.pth.tar".format(curr_lr)

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
        'state_dict': model_state_dict,
        'best_metric': current_metric,
        'optimizer' : optimizer.state_dict(),
        }, epoch_idx, is_best, filename=filename)
