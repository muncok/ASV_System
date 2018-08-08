# coding: utf-8
import torch

from train.train_utils import (set_seed, find_optimizer, get_dir_path,
load_checkpoint, save_checkpoint, new_exp_dir)
from data.dataloader import init_loaders
from data.data_utils import find_dataset, find_trial
from utils.parser import (train_parser, set_train_config)
from model.model_utils import find_model

from tensorboardX import SummaryWriter
from train.si_train import train, val, sv_test


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

#########################################
# Model Save Path
#########################################
if config['input_file']:
    # start new experiment continuing from "input_file"
    load_checkpoint(config, model, criterion, optimizer)
    config['output_dir'] = new_exp_dir(config,
            get_dir_path(config['input_file'])[:-4])
else:
    # start new experiment
    config['output_dir'] = new_exp_dir(config)

print("Model will be saved to : {}".format(config['output_dir']))

#########################################
# Model Training
#########################################
config['print_step'] = 100
set_seed(config)

# log configuration
log_dir = config['output_dir']
writer = SummaryWriter(log_dir)

# dataloader and scheduler
train_loader, val_loader, test_loader, sv_loader = loaders
# trial = pd.read_pickle("dataset/dataframes/voxc/voxc_trial.pkl")
trial = find_trial(config)

min_eer = config['best_metric'] if 'best_metric' in config else 1.0

for epoch_idx in range(config["s_epoch"], config["n_epochs"]):

    curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
    idx = 0
    while(epoch_idx >= config['lr_schedule'][idx] and
            idx < len(config['lr_schedule'])):
        # use new lr from schedule epoch not a next epoch
        idx += 1
    curr_lr = config['lrs'][idx]
    optimizer.state_dict()['param_groups'][0]['lr'] = curr_lr
    print("curr_lr: {}".format(curr_lr))

    train_loss, train_acc = train(config, train_loader, model, optimizer, criterion)
    val_loss, val_acc = val(config, val_loader, model, criterion)
    eer, label, score = sv_test(config, sv_loader, model, trial)

    writer.add_scalar("train/lr", curr_lr, epoch_idx)
    writer.add_scalar('train/loss', train_loss, epoch_idx)
    writer.add_scalar('train/acc', train_acc, epoch_idx)
    writer.add_scalar('val/loss', val_loss, epoch_idx)
    writer.add_scalar('val/acc', val_acc, epoch_idx)
    writer.add_scalar('sv_eer', eer, epoch_idx)
    writer.add_pr_curve('DET', label, score, epoch_idx)
    # for name, param in model.named_parameters():
        # writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx)

    print("epoch #{}, val accuracy: {}".format(epoch_idx, val_acc))
    print("epoch #{}, sv eer: {}".format(epoch_idx, eer))

    if eer < min_eer:
        min_eer = eer
        is_best = True
    else:
        is_best = False

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
        'loss': config["loss"],
        'state_dict': model_state_dict,
        'best_metric': eer,
        'optimizer' : optimizer.state_dict(),
        }, epoch_idx, is_best, filename=filename)

#########################################
# Model Evaluation
#########################################
test_loss, test_acc = val(config, test_loader, model, criterion)
