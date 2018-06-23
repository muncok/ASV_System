import numpy as np
import pandas as pd
import pickle

import scipy.stats as st
from sklearn.metrics import roc_curve
from tqdm import tqdm

import torch
from torch.autograd import Variable

def recToEER(eer_record, thresh_record, verbose=False):
    mean_eer = np.mean(eer_record)
    if verbose:
        lb, ub = st.t.interval(0.95, len(eer_record) - 1, loc=mean_eer, scale=st.sem(eer_record))
        # mean_thresh = np.mean(thresh_record)
        # lb_th, ub_th = st.t.interval(0.95, len(thresh_record) - 1, loc=mean_thresh, scale=st.sem(thresh_record))
        return (mean_eer, ub-mean_eer)
    return mean_eer

def lda_on_tensor(tensor, lda):
    return torch.from_numpy(lda.transform(tensor.numpy()).astype(np.float32))

def embeds_utterance(config, val_dataloader, model, lda=None):
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    val_iter = iter(val_dataloader)
    model.eval()
    splice_dim = config['splice_frames']
    embeddings = []
    labels = []
    for batch in tqdm(val_iter, total=len(val_iter)):
        x, y = batch
        time_dim = x.size(2)
        split_points = range(0, time_dim-(splice_dim+12), splice_dim)
        model_outputs = []
        for point in split_points:
            x_in = Variable(x.narrow(2, point, splice_dim+12))
            if not config['no_cuda']:
                x_in = x_in.cuda()
            model_outputs.append(model.embed(x_in).cpu().data)
        model_output = torch.stack(model_outputs, dim=0)
        model_output = model_output.mean(0)
        if lda is not None:
            model_output = torch.from_numpy(lda.transform(model_output.numpy()).astype(np.float32))
        embeddings.append(model_output)
        labels.append(y.numpy())
    embeddings = torch.cat(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels

def compute_cordination(trn, ndx):
    x_cord = []
    y_cord = []
    ndx_file =pd.DataFrame(ndx.file.unique().tolist(), columns=['file'])
    all_trials = trn.id.unique().tolist()
    for trial_id in tqdm(all_trials):
        trial_ndx = ndx[(ndx.id == trial_id)].reset_index()
        trial_embed_idx = np.nonzero(ndx_file.file.isin(trial_ndx.file))[0].tolist()
        x_cord += [all_trials.index(trial_id)] * len(trial_embed_idx)
        y_cord += trial_embed_idx

    cord = [x_cord, y_cord]
    pickle.dump(cord, open("../dataset/dataframes/reddots/m_part1/ndx_idxs.pkl", "wb"))


def decision_cost(sim_array, ths, labels, cost_miss=10, cost_fa=1, p_target=0.01):
    decision = sim_array > ths
    incorrect_decisions = (decision != labels)
    non_targets = (labels == 0)
    targets = (labels == 1)
    fa_rate = np.sum(incorrect_decisions[non_targets]) / np.sum(decision)
    miss_rate = np.sum(incorrect_decisions[targets]) / np.sum(~decision)
    cost = cost_miss * miss_rate * p_target + cost_fa * fa_rate * (1-p_target)
    return cost

def compute_eer(pos_scores, neg_scores):
    score_vector = np.concatenate([pos_scores, neg_scores])
    label_vector = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
    eer = np.min([fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))],
                 1-tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]])
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    print("eer:{:.3f}, thres:{:.4f}".format(eer*100, thres))

