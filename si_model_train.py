# coding: utf-8
import os

from data.data_utils import split_df, find_dataset
from data.dataloader import init_loaders_from_df
from model.model_utils import find_model
from utils.parser import (default_config, train_parser, set_config)
import train.si_train_v0 as si_train_v0
import train.si_train_v1 as si_train_v1
from train.train_utils import find_criterion, set_seed

#########################################
# Parser
#########################################
parser = train_parser()
args = parser.parse_args()
model = args.model
dataset = args.dataset
train_ver = args.version

si_config = default_config(model)
si_config = set_config(si_config, args)

#########################################
# Dataset loaders
#########################################
df, dataset_class, n_labels = find_dataset(si_config, dataset)
split_dfs = split_df(df)
loaders = init_loaders_from_df(si_config, split_dfs, dataset_class)

#########################################
# Model Initialization
#########################################
si_model= find_model(si_config, model, n_labels)
criterion = find_criterion(si_config, model, n_labels)
print("Our model: {}".format(model))

#########################################
# Model Save Configuration
#########################################
if si_config['input_file']:
    # start with "input_file"
    si_model.load(si_config['input_file'])
    if args.start_epoch > 0:
    # continuing
        si_model['output_file'] = si_config['input_file']
        print("training start from {} epoch".format(args.start_epoch))
else:
    # starting
    output_dir = ("models/compare_train_methods/{dset}/"
            "{model}_{in_format}_{in_len}f_{s_len}f").format(
                    dset=dataset, model=model,
                    in_len=si_config["input_frames"],
                    s_len=si_config["splice_frames"],
                    in_format=si_config["input_format"])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # suffix: e1, e2 ...
    done = False
    e = 0
    while not done:
        output_file = "e{version:02d}{suffix}.pt".format(
                version=e,
                suffix=si_config['suffix'])
        si_config['output_file'] = os.path.join(output_dir, output_file)
        if not os.path.isfile(si_config['output_file']):
            done = True
        else:
            e += 1
# model initialization
print("Save to : {}".format(si_config['output_file']))

#########################################
# Model Training
#########################################
si_config['print_step'] = 100
si_config["seed"] = 1337
set_seed(si_config)
if train_ver == 0:
    si_train_v0.si_train(si_config, model=si_model, loaders=loaders,
            criterion=criterion)
elif train_ver == 1:
    si_train_v1.si_train(si_config, model=si_model, loaders=loaders,
            criterion=criterion)

#########################################
# Model Evaluation
#########################################
# si_train.evaluate(si_config, si_model, loaders[-1])
