# coding: utf-8
import os
import argparse
import numpy as np
import pandas as pd
import pickle

from model.model_utils import find_model
from system.si_train import find_criterion, load_checkpoint
from system.sv_test import get_embeds, sv_test
from system.sv_test import compute_minDCF
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
                    help='directory for embeds to be saved',)

# parser.add_argument('-n_labels',
                    # type=int,
                    # help='n_labels of input_model',
                    # default=None)

args = parser.parse_args()
config = vars(args)
config['input_clip'] = False
config['gpu_no'] = [0]
config['no_cuda'] = not args.cuda

#########################################
# Dataset Load
#########################################
# dataset = config['dataset']
# name, in_format, in_dim, mode = dataset.split("_")
config['data_folder'] = "datasets/voxceleb12/feats/fbank64_vad"
config['input_format'] = 'fbank'
config['input_dim'] = 64
config['n_labels'] = 1211
df = pd.read_csv("datasets/voxceleb1/dataframes/voxc1_sv.csv")
dataset = FeatDataset.read_df(config, df, "test")
sv_loader = init_default_loader(
        config, dataset, shuffle=False,
        collate_fn=_var_len_collate_fn)

#########################################
# Model Initialization
#########################################
model = find_model(config)
load_checkpoint(config, model)
criterion = find_criterion(config, model)

#########################################
# Scoring
#########################################
# sv eer: equal error rate on sv(speaker verification) dataset
trial = pd.read_csv("datasets/voxceleb1/dataframes/voxc1_sv_trial.csv")
sv_embeddings = get_embeds(config, sv_loader, model)
eer, thres, scores = sv_test(sv_embeddings, trial)
print("sv eer: {:.4f} (model:{}, dataset:voxc1)".format(
    eer, config['arch']))
compute_minDCF(scores, trial.label.tolist())

# if config['output_dir']:
    # output_dir = config['output_dir']
    # if not os.path.isdir(output_dir):
        # os.makedirs(output_dir)
    # sv_keys = df.id.tolist()
    # pickle.dump(sv_keys, open(output_dir+"/voxc12_keys.pkl", "wb"))
    # np.save(output_dir+"/voxc12_embeds.npy", sv_embeddings.cpu().numpy())
    # print("embeds are saved at {}".format(output_dir))
