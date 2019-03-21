# coding: utf-8
import os
import pickle
import numpy as np
import argparse
# import ipdb

from data.dataloader import init_default_loader
from data.dataloader import _var_len_collate_fn
from data.data_utils import find_dataset
from model.model_utils import find_model
from system.si_train import load_checkpoint
from system.sv_test import extract_features_var_len
#########################################
# Parser
#########################################
parser = argparse.ArgumentParser()
parser.add_argument('-batch', '--batch_size',
                    type=int,
                    help='batch size',
                    default=64)

parser.add_argument('-n_workers', '--num_workers',
                    type=int,
                    help='number of workers of dataloader',
                    default=0)

parser.add_argument('-dataset',
                    type=str,
                    required=True,
                    help='{name}_{format}_{dim}_{wav|feat}')

parser.add_argument('-arch',
                    type=str,
                    required=True,
                    help='type of model')

parser.add_argument('-input_file',
                    type=str,
                    required=True,
                    help='model path to be loaded')

parser.add_argument('-cuda',
                    action = 'store_true',
                    default= False)

parser.add_argument('-output_dir',
                    type=str,
                    required=True,
                    help='path for embeds to be saved',)

parser.add_argument('-n_labels',
                    type=int,
                    help='n_labels of input_model',
                    default=None)

args = parser.parse_args()
config = vars(args)
config['input_clip'] = False
config['gpu_no'] = [0]
config['no_cuda'] = not args.cuda
config['no_eer'] = False # TODO: for sv_set, we should set no_eer as False
config['random_clip'] = False

output_dir = args.output_dir
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

#########################################
# Dataset & Model Initialization
#########################################
# split=False ==> si_set, sv_set
dfs, datasets = find_dataset(config, split=True)
si_train_df, si_val_df, sv_df = dfs
si_train_dset, si_val_dset, sv_dset = datasets

model = find_model(config)
load_checkpoint(config, model=model)

#########################################
# Compute Test Embeddings
#########################################
# val_dataloader = init_default_loader(config, si_val_dset, shuffle=False,
        # collate_fn=_var_len_collate_fn)
# val_embeddings = extract_features_var_len(config, val_dataloader, model)
# val_keys = si_val_df.id.tolist()
# pickle.dump(val_keys, open(output_dir+"/val_keys.pkl", "wb"))
# pickle.dump(val_embeddings, open(output_dir+"/val_embeds.pkl", "wb"))

#########################################
# Compute SV Embeddings
#########################################
sv_dataloader = init_default_loader(config, sv_dset, shuffle=False,
        collate_fn=_var_len_collate_fn)
sv_embeddings = extract_features_var_len(config, sv_dataloader, model)
sv_keys = sv_df.id.tolist()
pickle.dump(sv_keys, open(output_dir+"/sv_keys.pkl", "wb"))
pickle.dump(sv_embeddings, open(output_dir+"/sv_embeds.pkl", "wb"))
