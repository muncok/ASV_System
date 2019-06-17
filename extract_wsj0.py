# coding: utf-8
import os
import pickle
import numpy as np
import pandas as pd
import argparse
# import ipdb

from data.dataloader import init_default_loader
from data.feat_dataset import FeatDataset
from model.model_utils import find_model
from system.si_train import load_checkpoint
from system.sv_test import extract_embed_var_len
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

parser.add_argument('-output_dir',
                    type=str,
                    required=True,
                    help='path for embeds to be saved',)

parser.add_argument('-n_labels',
                    type=int,
                    required=True,
                    help='n_labels of input_model',
                    default=None)

args = parser.parse_args()
config = vars(args)
config['input_clip'] = False
config['gpu_no'] = [0]
config['no_cuda'] = not args.cuda
config['no_eer'] = False # TODO: for sv_set, we should set no_eer as False
config['random_clip'] = False


#########################################
# Dataset & Model Initialization
#########################################
# split=False ==> si_set, sv_set
df = pd.read_pickle(config['df'])
config['data_folder'] = "/dataset/SV_sets/wsj0/fbank64/"
config['input_dim'] = 64
config['n_labels'] = args.n_labels
dataset = FeatDataset.read_df(config, df, 'test')
model = find_model(config)
load_checkpoint(config, model=model)

#########################################
# Extract Embeddings
#########################################
val_dataloader = init_default_loader(config, dataset, shuffle=False,
        var_len=True)
val_embeddings, _ = extract_embed_var_len(config, val_dataloader, model)

if config['output_dir']:
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    sv_keys = df.index.tolist()
    pickle.dump(sv_keys, open(output_dir+"/sv_keys.pkl", "wb"))
    np.save(output_dir+"/sv_embeds.npy", val_embeddings.cpu().numpy())
    print("embeds are saved at {}".format(output_dir))
