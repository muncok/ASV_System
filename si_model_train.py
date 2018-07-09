# coding: utf-8
import os

from utils.parser import (default_config, train_parser, set_config)
from train import (si_train_v0, si_train_v1, si_train_v2)
from data.dataloader import init_loaders_from_df
from train.train_utils import set_seed, find_optimizer, get_dir_path
from model.model_utils import find_model
from data.data_utils import split_df, find_dataset

def new_exp_dir(old_exp_dir):
    # suffix: v1, v2 ...
    done = False
    v = 0
    while not done:
        output_dir_ = "{output_dir}_v{version:02d}".format(
                output_dir=old_exp_dir, version=v)
        if not os.path.isdir(output_dir_):
            output_dir = output_dir_
            os.makedirs(output_dir)
            done = True
        else:
            v += 1
    output_file = os.path.join(output_dir, "model.pt")
    return output_file

#########################################
# Parser
#########################################
parser = train_parser()
args = parser.parse_args()
arch = args.arch
dataset = args.dataset
train_ver = args.version

config = default_config(arch)
config = set_config(config, args, 'train')

#########################################
# Dataset loaders
#########################################
df, dataset_type, n_labels = find_dataset(config)
split_dfs = split_df(df)
loaders = init_loaders_from_df(config, split_dfs, dataset_type)

#########################################
# Model Initialization
#########################################
model= find_model(config, n_labels)
criterion, optimizer = find_optimizer(config, model)

#########################################
# Model Save Path
#########################################
if config['input_file']:
    # start new experiment continuing from "input_file"
    config['output_file'] = new_exp_dir(get_dir_path(config['input_file']))
else:
    # start new experiment
    new_output_dir = ("models/compare_train_methods/{dset}/"
            "{arch}_{loss}/{in_format}_{in_len}f_{s_len}f_{suffix}").format(
                    dset=dataset, arch=arch, loss=config["loss"],
                    in_len=config["input_frames"],
                    s_len=config["splice_frames"],
                    in_format=config["input_format"],
                    suffix=config["suffix"])
    config['output_file'] = new_exp_dir(new_output_dir)

print("Model will be saved to : {}".format(config['output_file']))

#########################################
# Model Training
#########################################
config['print_step'] = 100
config["seed"] = args.seed
set_seed(config)
if train_ver == 0:
    si_train_v0.si_train(config, model=model, loaders=loaders)
elif train_ver == 1:
    si_train_v1.si_train(config, model=model, loaders=loaders,
            optimizer=optimizer, criterion=criterion)
elif train_ver == 2:
    si_train_v2.si_train(config, model=model, loaders=loaders,
            optimizer=optimizer, criterion=criterion)

#########################################
# Model Evaluation
#########################################
# si_train.evaluate(config, model, loaders[-1])
