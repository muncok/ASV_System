# coding: utf-8
import os
import argparse
import numpy as np
import pandas as pd
import pickle

from model.model_utils import find_model
from system.si_train import find_criterion, load_checkpoint
from system.sv_test import get_embeds
from data.dataloader import init_default_loader, _var_len_collate_fn
from data.feat_dataset import FeatDataset

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

# parser.add_argument('-dataset',
                    # type=str,
                    # required=True,
                    # help='{name}_{format}_{dim}_{wav|feat}')

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
                    help='directory for embeds to be saved',)

parser.add_argument('-n_labels',
                    type=int,
                    help='n_labels of input_model',
                    default=None)

args = parser.parse_args()
config = vars(args)
config['input_clip'] = False
config['gpu_no'] = [0]
config['no_cuda'] = not args.cuda

#########################################
# Dataset Load
#########################################
dataset = 'voxc12aug_fbank_64_feat'
name, in_format, in_dim, mode = dataset.split("_")
config['data_folder'] = "/dataset/SV_sets/voxceleb12/feats/fbank64_vad"
config['input_format'] = in_format
config['input_dim'] = int(in_dim)
config['num_workers'] = 8
config['n_labels'] = 7365

si_df = pd.read_csv("/dataset/SV_sets/voxceleb12/dataframes/voxc12aug_si.csv")
si_dataset = FeatDataset.read_df(config, si_df, "test")
si_loader = init_default_loader(
        config, si_dataset, shuffle=False,
        collate_fn=_var_len_collate_fn)

sv_df = pd.read_csv("/dataset/SV_sets/voxceleb12/dataframes/voxc12aug_sv.csv")
sv_dataset = FeatDataset.read_df(config, sv_df, "test")
sv_loader = init_default_loader(
        config, sv_dataset, shuffle=False,
        collate_fn=_var_len_collate_fn)

#########################################
# Model Initialization
#########################################
model = find_model(config)
load_checkpoint(config, model)
criterion = find_criterion(config, model)

#########################################
# Compute Train Embeddings
#########################################
output_dir = args.output_dir
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# si_embeddings = get_embeds(config, si_loader, model)
# si_keys = si_df.id.tolist()
# pickle.dump(si_keys, open(os.path.join(output_dir, "si_keys.pkl"), "wb"))
# np.save(os.path.join(output_dir, "si_embeds.npy"), si_embeddings)

#########################################
# Compute Test Embeddings
#########################################

sv_embeddings = get_embeds(config, sv_loader, model)
sv_keys = sv_df.id.tolist()
pickle.dump(sv_keys, open(os.path.join(output_dir, "sv_keys.pkl"), "wb"))
np.save(os.path.join(output_dir, "sv_embeds.npy"), sv_embeddings)
