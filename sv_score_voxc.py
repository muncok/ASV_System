# coding: utf-8
import pandas as pd
import os

import torch.nn.functional as F
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from data.dataloader import init_default_loader
from model.model_utils import find_model
from utils.parser import default_config, score_parser, set_score_config
from data.data_utils import find_dataset
from sv_score.score_utils import embeds_utterance
from train.train_utils import load_checkpoint

def embed_path(model_path, file_name="embed.pkl"):
    dir_path = os.path.join(*model_path.split("/")[:-1])
    return os.path.join(dir_path, file_name)

#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
model = args.arch

si_config = default_config(model)
si_config = set_score_config(si_config, args)

#########################################
# Model Initialization
#########################################
# set data_folder, input_dim, n_labels, and dset
_, dset, n_labels = find_dataset(si_config)
si_model = find_model(si_config, n_labels)
load_checkpoint(si_config, si_model, None)
lda = None

#########################################
# Compute Embeddings
#########################################
voxc_test_df = pd.read_pickle("dataset/dataframes/voxc/sv_voxc_dataframe.pkl")
voxc_test_dset = dset.read_df(si_config, voxc_test_df, "test")
val_dataloader = init_default_loader(si_config, voxc_test_dset, shuffle=False)
embeddings, _ = embeds_utterance(si_config, val_dataloader, si_model, lda)

#########################################
# Load trial
#########################################
trial = pd.read_pickle("dataset/dataframes/voxc/voxc_trial.pkl")
sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
cord = [trial.enrolment_id.tolist(), trial.test_id.tolist()]
score_vector = sim_matrix[cord].numpy()
label_vector = trial.label
fpr, tpr, thres = roc_curve(
        label_vector, score_vector, pos_label=1)
# eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thres = interp1d(fpr, thres)(eer)
print("[TI] eer: {:.3f}%".format(eer*100))
