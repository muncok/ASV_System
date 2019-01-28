# coding: utf-8
from system.si_train import load_checkpoint
from data.data_utils import find_trial
from model.tdnn import tdnn_xvector_nofc
from utils.parser import score_parser, set_score_config
from data.dataloader import init_default_loader, _var_len_collate_fn
from system.sv_test import sv_test


#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)
from data.data_utils import find_dataset
config['no_eer'] = False
dfs, datasets = find_dataset(config, split=False)


#########################################
# Model Initialization
#########################################
config['no_eer'] = False
model = tdnn_xvector_nofc(config, 512, config['n_labels'])
if not config["no_cuda"]:
    model.cuda()
load_checkpoint(config, model)

#########################################
# sv eer: equal error rate on sv(speaker verification) dataset
import pandas as pd
from data.feat_dataset import FeatDataset
from system.sv_test import get_embeds

si_df = dfs[0]
si_dataset = datasets[0]
si_loader = init_default_loader(
        config, si_dataset, shuffle=False,
        collate_fn=_var_len_collate_fn)
si_embeddings = get_embeds(config, si_loader, model)

sv_df = dfs[1]
sv_dataset = datasets[1]
sv_loader = init_default_loader(
        config, sv_dataset, shuffle=False,
        collate_fn=_var_len_collate_fn)
trial = find_trial(config)
sv_embeddings = get_embeds(config, sv_loader, model)

eer, _, _ = sv_test(sv_embeddings, trial)
print("sv eer: {:.4f} (model:{}, dataset:{})".format(eer, config['arch'], config['dataset']))

import os
import pickle
import numpy as np
if config['output_dir']:
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    si_keys = si_df.id.tolist()
    pickle.dump(si_keys, open(output_dir+"/si_keys.pkl", "wb"))
    np.save(output_dir+"/si_embeds.npy", si_embeddings.cpu().numpy())
    sv_keys = sv_df.id.tolist()
    pickle.dump(sv_keys, open(output_dir+"/sv_keys.pkl", "wb"))
    np.save(output_dir+"/sv_embeds.npy", sv_embeddings.cpu().numpy())
    print("embeds are saved at {}".format(output_dir))
