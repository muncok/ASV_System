# coding: utf-8
import os
import pickle
import numpy as np

from data.dataloader import init_default_loader
from utils.parser import score_parser, set_score_config
from data.data_utils import find_dataset
from eval.sv_test import embeds_sub_utterance
from train.train_utils import load_checkpoint, get_dir_path
from model.model_utils import find_model

#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)

if not config['output_dir']:
    output_dir = get_dir_path(config['input_file'])
    output_dir = os.path.join(output_dir, 'embeds')
else:
    output_dir = config["output_dir"]

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

if not config['lda_file']:
    lda = None
else:
    lda = pickle.load(open(config['lda_file'], "rb"))

#########################################
# Compute Train Embeddings
#########################################
si_dataloader = init_default_loader(config, si_dset, shuffle=False)
si_embeddings, _ = embeds_sub_utterance(config, si_dataloader, model, lda)

si_keys = si_df.index.tolist()
pickle.dump(si_keys, open(os.path.join(output_dir, "si_keys.pkl"), "wb"))
np.save(os.path.join(output_dir, "si_embeds.npy"), si_embeddings)

#########################################
# Compute Test Embeddings
#########################################
sv_dataloader = init_default_loader(config, sv_dset, shuffle=False)
sv_embeddings, _ = embeds_sub_utterance(config, sv_dataloader, model, lda)

sv_keys = sv_df.index.tolist()
pickle.dump(sv_keys, open(os.path.join(output_dir, "sv_keys.pkl"), "wb"))
np.save(os.path.join(output_dir, "sv_embeds.npy"), sv_embeddings)
