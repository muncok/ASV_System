# coding: utf-8
from train.train_utils import find_criterion, load_checkpoint
from data.data_utils import find_dataset, find_trial
from utils.parser import score_parser, set_score_config
from data.dataloader import init_default_loader
from eval.sv_test import sv_test
from model.model_utils import find_model


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
model, _ = load_checkpoint(config, model)
criterion = find_criterion(config, model)

#########################################
# Model Training
sv_set = datasets[-1]
sv_loader = init_default_loader(config, sv_set, shuffle=False)
trial = find_trial(config)
eer, label, score = sv_test(config, sv_loader, model, trial)
print("sv eer: {:.4f} (model:{}, dataset:{})".format(eer, config['arch'], config['dataset']))

