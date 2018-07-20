# coding: utf-8
import os
import pandas as pd
import pickle

from data.dataloader import init_default_loader
from utils.parser import score_parser, set_score_config
from data.data_utils import find_dataset
from sv_score.score_utils import embeds_utterance
from train.train_utils import load_checkpoint, get_dir_path

#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)
output_folder = get_dir_path(config['input_file'])

#########################################
# Model Initialization
#########################################
# set data_folder, input_dim, n_labels, and dset
_, dset = find_dataset(config)
model, _ = load_checkpoint(config)
lda = None

#########################################
# Compute Train Embeddings
#########################################
voxc_train_df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
voxc_train_dset = dset.read_df(config, voxc_train_df, "train")
val_dataloader = init_default_loader(config, voxc_train_dset, shuffle=False)
embeddings, _ = embeds_utterance(config, val_dataloader, model, lda)

dvec_dict = dict(zip(voxc_train_df.index.tolist(),
    embeddings.numpy()))

pickle.dump(dvec_dict, open(os.path.join(output_folder,
    "voxc_train_dvectors.pkl"), "wb"))

#########################################
# Compute Test Embeddings
#########################################
voxc_test_df = pd.read_pickle("dataset/dataframes/voxc/sv_voxc_dataframe.pkl")
voxc_test_dset = dset.read_df(config, voxc_test_df, "test")
val_dataloader = init_default_loader(config, voxc_test_dset, shuffle=False)
embeddings, _ = embeds_utterance(config, val_dataloader, model, lda)

dvec_dict = dict(zip(voxc_test_df.index.tolist(),
    embeddings.numpy()))
pickle.dump(dvec_dict, open(os.path.join(output_folder,
    "voxc_test_dvectors.pkl"), "wb"))
