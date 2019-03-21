# coding: utf-8
import os
import pickle
import numpy as np
import argparse

from data.dataloader import init_default_loader
from data.data_utils import find_dataset
from system.sv_test import extract_embed_var_len
from system.si_train import load_checkpoint
from model.model_utils import find_model
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
dfs, datasets = find_dataset(config, split=False)
si_df, sv_df = dfs
si_dset, sv_dset = datasets

model = find_model(config)
load_checkpoint(config, model=model)

#########################################
# Compute Train Embeddings
#########################################
si_dataloader = init_default_loader(config, si_dset, shuffle=False,
        var_len=True)
si_embeddings, _ = extract_embed_var_len(config, si_dataloader, model)
si_keys = si_df.id.tolist()
pickle.dump(si_keys, open(os.path.join(output_dir, "si_keys.pkl"), "wb"))
np.save(os.path.join(output_dir, "si_embeds.npy"), si_embeddings)

#########################################
# Compute Test Embeddings
#########################################

sv_dataloader = init_default_loader(config, sv_dset, shuffle=False,
        var_len=True)
sv_embeddings, _ = extract_embed_var_len(config, sv_dataloader, model)
sv_keys = sv_df.id.tolist()
pickle.dump(sv_keys, open(os.path.join(output_dir, "sv_keys.pkl"), "wb"))
np.save(os.path.join(output_dir, "sv_embeds.npy"), sv_embeddings)
