# coding: utf-8
import os
import pickle
import numpy as np

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
if config['output_folder'] is None:
    output_folder = get_dir_path(config['input_file'])
else:
    output_folder = config["output_folder"]

#########################################
# Model Initialization
#########################################
# set data_folder, input_dim, n_labels, and dset
dfs, datasets = find_dataset(config, split=False)
si_df, sv_df = dfs
si_dset, sv_dset = datasets
model, _ = load_checkpoint(config)
lda = None

#########################################
# Compute Train Embeddings
#########################################
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

val_dataloader = init_default_loader(config, si_dset, shuffle=False)
embeddings, _ = embeds_utterance(config, val_dataloader, model, lda)
si_keys = si_df.index.tolist()
si_embeds = embeddings.numpy()

pickle.dump(si_keys, open(os.path.join(output_folder, "si_keys.pkl"), "wb"))
np.save(si_embeds, open(os.path.join(output_folder, "si_embeds.npy"), "wb"))

#########################################
# Compute Test Embeddings
#########################################
val_dataloader = init_default_loader(config, sv_dset, shuffle=False)
embeddings, _ = embeds_utterance(config, val_dataloader, model, lda)
sv_keys = sv_df.index.tolist()
sv_embeds = embeddings.numpy()

pickle.dump(sv_keys, open(os.path.join(output_folder, "sv_keys.pkl"), "wb"))
np.save(sv_embeds, open(os.path.join(output_folder, "sv_embeds.npy"), "wb"))
