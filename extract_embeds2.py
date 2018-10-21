# coding: utf-8
import os
import pickle
import numpy as np

from data.dataloader import init_default_loader
from utils.parser import score_parser, set_score_config
from data.data_utils import find_dataset
from eval.sv_test import embeds_utterance_frames
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

#########################################
# Compute Train Embeddings
#########################################
si_dataloader = init_default_loader(config, si_dset, shuffle=False)
si_embeddings, _ = embeds_utterance_frames(config, config['input_frames'], si_dataloader, model)

si_keys = si_df.index.tolist()
pickle.dump(si_keys, open(os.path.join(output_dir, "si_keys.pkl"), "wb"))
np.save(os.path.join(output_dir, "si_embeds.npy"), si_embeddings)

#########################################
# Compute Test Embeddings
#########################################
sv_dataloader = init_default_loader(config, sv_dset, shuffle=False)
enroll_spFr, test_spFr = config['splice_frames']
sv_enroll_embeddings, _ = embeds_utterance_frames(config, enroll_spFr, sv_dataloader, model)

if enroll_spFr == test_spFr:
    sv_test_embeddings = sv_enroll_embeddings
else:
    sv_test_embeddings, _ = embeds_utterance_frames(config, test_spFr, sv_dataloader, model)

sv_keys = sv_df.index.tolist()
pickle.dump(sv_keys, open(os.path.join(output_dir, "sv_keys.pkl"), "wb"))
np.save(os.path.join(output_dir, "sv_enroll_embeds.npy"), sv_enroll_embeddings)
np.save(os.path.join(output_dir, "sv_test_embeds.npy"), sv_test_embeddings)
