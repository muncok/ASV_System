# coding: utf-8
import pandas as pd

import torch.nn.functional as F

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from data.dataloader import init_default_loader
from data.data_utils import find_dataset
from utils.parser import score_parser, set_score_config
from sv_score.score_utils import embeds_utterance
from train.train_utils import load_checkpoint

#########################################
# Parser
#########################################
parser = score_parser()
args = parser.parse_args()
config = set_score_config(args)

#########################################
# Model Initialization
#########################################
_, dset = find_dataset(config)
model, _ = load_checkpoint(config)
lda = None

#########################################
# Compute Embeddings
#########################################
test_df = pd.read_pickle(
        "dataset/dataframes/gcommand/equal_num_102spk/equal_num_102spk_sv.pkl")
test_dset = dset.read_df(config, test_df, "test")
val_dataloader = init_default_loader(config, test_dset, shuffle=False)
embeddings, _ = embeds_utterance(config, val_dataloader, model, lda)

#########################################
# Load trial
#########################################
trial = pd.read_pickle(
        "dataset/dataframes/gcommand/equal_num_102spk/equal_num_102spk_trial.pkl")
sim_matrix = F.cosine_similarity(
        embeddings.unsqueeze(1),
        embeddings.unsqueeze(0),
        dim=2)
cord = [trial.enrolment_id.tolist(), trial.test_id.tolist()]
score_vector = sim_matrix[cord].numpy()
label_vector = trial.label
fpr, tpr, thres = roc_curve(
        label_vector, score_vector, pos_label=1)
# eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
thres = interp1d(fpr, thres)(eer)
print("[TI] eer: {:.3f}%".format(eer*100))
