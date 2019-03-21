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

parser.add_argument('-df',
                    type=str,
                    required=True,
                    help='dataframe file')

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

parser.add_argument('-output_file',
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

output_dir = "/".join(args.output_file.split("/")[:-1])
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

#########################################
# Dataset & Model Initialization
#########################################
# split=False ==> si_set, sv_set
import pandas as pd
from data.feat_dataset import FeatDataset
df = pd.read_csv(config['df'])
config['data_folder'] = "/dataset/SV_sets/voxceleb12/feats/fbank64_vad"
config['input_dim'] = 64
config['n_labels'] = 1211
dataset = FeatDataset.read_df(config, df, 'test')
model = find_model(config)
load_checkpoint(config, model=model)

#########################################
# Extract Embeddings
#########################################
val_dataloader = init_default_loader(config, dataset, shuffle=False,
        collate_fn=_var_len_collate_fn)
val_embeddings = extract_features_var_len(config, val_dataloader, model)
pickle.dump(val_embeddings, open(args.output_file, "wb"))

