# coding: utf-8
import pandas as pd
import numpy as np
import pickle

import torch.nn.functional as F
from sklearn.metrics import roc_curve

from sv_system.data.dataloader import init_default_loader
from sv_system.model import find_model
from sv_system.utils.parser import default_config, train_parser, set_input_config, set_train_config
from sv_system.data.dataset import mfccDataset
from score_utils import embeds_utterance

#########################################
# Parser
#########################################
parser = train_parser()
args = parser.parse_args()
model = args.model
dataset = args.dataset

si_config = default_config(model)
si_config = set_input_config(si_config, args)
si_config = set_train_config(si_config, args)

#########################################
# Model Initialization
#########################################
si_config["input_dim"] = 20
model, criterion = find_model(si_config, model, 1)
# model.load("models/compare_train_methods/reddots/si_reddots_TdnnModel_3s_0.1s_fbank_full_frame.pt")
lda = None

#########################################
# Compute Embeddings
#########################################
voxc_test_df = pd.read_pickle("dataset/dataframes/sv_voxc_dataframe.pkl")
si_config["data_folder"] = "dataset/voxceleb/mfcc"
voxc_test_dset = mfccDataset.read_df(si_config, voxc_test_df, "test")
val_dataloader = init_default_loader(si_config, voxc_test_dset, shuffle=False)
embeddings, _ = embeds_utterance(si_config, val_dataloader, model, lda)
# pickle.dump(embeddings, open("embed.pkl", "wb"))
# embeddings = pickle.load(open("embed.pkl", "rb"))
# print(embeddings.shape)

#########################################
# Load trial
#########################################
trial = pd.read_pickle("dataset/dataframes/voxc_trial.pkl")
sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
cord = [trial.enrolment_id.tolist(), trial.test_id.tolist()]
score_vector = sim_matrix[cord]
label_vector = trial.label.tolist()
fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
print("[TI] eer: {:.2f}".format(eer))
