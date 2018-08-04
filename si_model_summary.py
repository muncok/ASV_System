# coding: utf-8
import numpy as np
# from train.train_utils import  load_checkpoint
from model.model_utils import find_model
from utils.parser import score_parser, set_score_config
# from data.dataloader import init_default_loader
# from train.si_train import val


#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)

#########################################
# Model Initialization
#########################################
config['n_labels'] = 1210
model = find_model(config)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

layer_parameters = filter(lambda p: p.requires_grad, model.conv1.parameters())
conv1_params = sum([np.prod(p.size()) for p in layer_parameters])
print(conv1_params)

layer_parameters = filter(lambda p: p.requires_grad, model.layer1.parameters())
conv1_params = sum([np.prod(p.size()) for p in layer_parameters])
print(conv1_params)

layer_parameters = filter(lambda p: p.requires_grad, model.layer2.parameters())
conv1_params = sum([np.prod(p.size()) for p in layer_parameters])
print(conv1_params)

layer_parameters = filter(lambda p: p.requires_grad, model.layer3.parameters())
conv1_params = sum([np.prod(p.size()) for p in layer_parameters])
print(conv1_params)

layer_parameters = filter(lambda p: p.requires_grad, model.layer4.parameters())
conv1_params = sum([np.prod(p.size()) for p in layer_parameters])
print(conv1_params)

layer_parameters = filter(lambda p: p.requires_grad, model.fc.parameters())
conv1_params = sum([np.prod(p.size()) for p in layer_parameters])
print(conv1_params)

layer_parameters = filter(lambda p: p.requires_grad, model.output.parameters())
conv1_params = sum([np.prod(p.size()) for p in layer_parameters])
print(conv1_params)
#########################################
