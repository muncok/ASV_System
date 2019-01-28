# coding: utf-8
import os
import sys
# import uuid
from system.si_train import print_eval
import numpy as np

import torch
from tensorboardX import SummaryWriter

from utils.parser import train_parser, set_train_config
from utils.utils import Logger

# from data.dataloader import init_loaders
from data.data_utils import find_dataset, find_trial

from system.si_train import set_seed
from system.si_train import load_checkpoint, save_checkpoint, new_exp_dir
from system.sv_test import sv_test
# from system.si_train import train, val
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.optim.lr_scheduler import MultiStepLR

from model.tdnn import tdnn_xvector_nofc
from loss.hist_loss import HistogramLoss


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
config['no_eer'] = False
dfs, datasets = find_dataset(config)
# loaders = init_loaders(config, datasets)

#########################################
# Model Initialization
#########################################
model = tdnn_xvector_nofc(config, 512, config['n_labels'])
if not config["no_cuda"]:
    model.cuda()
criterion = HistogramLoss(num_steps=251, cuda=config['cuda'])
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
    model.parameters()), lr=config['lrs'][0])
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
from data.samplers import HistSampler
train_sampler = HistSampler(dfs[0].label, config['batch_size'])
train_loader = DataLoader(datasets[0], batch_sampler=train_sampler,
        num_workers=config['num_workers'])
val_sampler = HistSampler(dfs[1].label, config['batch_size'])
val_loader = DataLoader(datasets[1], batch_sampler=val_sampler,
        num_workers=config['num_workers'])

#########################################
# trial
#########################################
trial = find_trial(config)
best_metric = 0.0

#########################################
# Model Training
#########################################
set_seed(config)
for epoch_idx in range(config["s_epoch"], config["n_epochs"]):
    config['epoch_idx'] = epoch_idx
    curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
    print("===============================================")
    print("epoch #{}".format(epoch_idx))
    print("curr_lr: {}".format(curr_lr))

    # train code
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
        embeds = model(X)
        loss = criterion(embeds, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        # schedule over iteration
        accs.append(print_eval("train step #{}".format('0'), embeds, y,
            loss_sum/(batch_idx+1), display=False))

        del embeds
        del loss
        if batch_idx in print_steps:
            print("[{}/{}] train, loss: {:.4f}, acc: {:.5f} ".format(
                batch_idx, len(train_loader),
                loss_sum/(batch_idx+1), np.mean(accs)))

    avg_acc = np.mean(accs)
    train_loss = loss_sum
    train_acc = avg_acc
    writer.add_scalar("train/lr", curr_lr, epoch_idx)
    writer.add_scalar('train/loss', train_loss, epoch_idx)
    writer.add_scalar('train/acc', train_acc, epoch_idx)
    plateau_scheduler.step(train_loss)

    # eer, label, score = sv_test(config, sv_loader, model, trial)
    # current_metric = eer
    # writer.add_scalar('sv_eer', eer, epoch_idx)
    # writer.add_pr_curve('DET', label, score, epoch_idx)
    # print("sv eer: {}".format(eer))
    # if eer < best_metric:
        # best_metric = eer
        # is_best = True
    # else:
        # is_best = False


    # for name, param in model.named_parameters():
        # writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx)

    filename = config["output_dir"] + "/model.{:.4}.pth.tar".format(curr_lr)

    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    save_checkpoint({'epoch': epoch_idx,
                    'step_no': (epoch_idx+1) * len(train_loader),
                    'arch': config["arch"],
                    'n_labels': config["n_labels"],
                    'dataset': config["dataset"],
                    'state_dict': model_state_dict,
                    'best_metric': 0,
                    'optimizer' : optimizer.state_dict()},
                    epoch_idx, True, filename=filename)
