# coding: utf-8
from system.si_train import find_criterion, load_checkpoint
from data.data_utils import find_dataset, find_trial
from model.model_utils import find_model
from utils.parser import score_parser, set_score_config
from data.dataloader import init_default_loader
from system.si_train import val
from system.sv_test import sv_test


#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)

#########################################
# Model Initialization
#########################################
_, datasets = find_dataset(config)
model = find_model(config)
load_checkpoint(config, model)
criterion = find_criterion(config, model)

#########################################
# val acc: model accuracy on si(speaker identification) dataset
# sv eer: equal error rate on sv(speaker verification) dataset
val_set = datasets[1]
val_loader = init_default_loader(config, val_set, shuffle=False)
val_loss, val_acc = val(config, val_loader, model, criterion)
print("val acc: {}".format(val_acc))

sv_set = datasets[-1]
sv_loader = init_default_loader(config, sv_set, shuffle=False)
trial = find_trial(config)
eer, _, _ = sv_test(config, sv_loader, model, trial)
print("sv eer: {:.4f} (model:{}, dataset:{})".format(eer, config['arch'], config['dataset']))
