import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

import torch

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
    val_iter = iter(val_dataloader)
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_iter, total=len(val_iter), ncols=100):
            x, y = batch
            if not config['no_cuda']:
                x= x.cuda()

            if config['score_mode'] == "precise":
                model_output = model.embed(x).cpu().data
            else:
                model_outputs = []
                time_dim = x.size(2)
                splice_dim = config['splice_frames']
                split_points = range(0, time_dim-(splice_dim)+1, 1)
                for point in split_points:
                    x_in = x.narrow(2, point, splice_dim)
                    model_outputs.append(model.embed(x_in).cpu().data)
                model_output = torch.stack(model_outputs, dim=0)
                model_output = model_output.mean(0)

            if lda is not None:
                model_output = torch.from_numpy(
                        lda.transform(model_output.numpy()).astype(np.float32))
            embeddings.append(model_output)
            labels.append(y.numpy())
        embeddings = torch.cat(embeddings)
        labels = np.hstack(labels)
    return embeddings, labels

def embeds_utterance1(config, val_dataloader, model, lda=None):
    val_iter = iter(val_dataloader)
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(val_iter, total=len(val_iter), ncols=100):
            x, y = batch
            x_in = x
            if not config['no_cuda']:
                x_in = x_in.cuda()
            model_output = model.embed(x_in).cpu().data
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
    # eer = np.min([fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))],
                 # 1-tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]])
    # thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thres = interp1d(fpr, thres)(eer)
    print("eer:{:.3f}, thres:{:.4f}".format(eer*100, thres))


def plot_eer(score_vector, label_vector):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    eer = dict()
    thres = dict()
    n_classes = score_vector.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], thres[i] = roc_curve(label_vector[:, i], score_vector[:, i], pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], thres["micro"] = roc_curve(label_vector.ravel(), score_vector.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    eer["micro"] = fpr["micro"][np.nanargmin(np.abs(fpr["micro"] - (1 - tpr["micro"])))]

    for i in range(n_classes):
        eer[i] = fpr[i][np.nanargmin(np.abs(fpr[i] - (1 - tpr[i])))]

    from scipy import interp
    from itertools import cycle
    lw = 2
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    eer["macro"] = fpr["macro"][np.nanargmin(np.abs(fpr["macro"] - (1 - tpr["macro"])))]
    return eer["micro"]
    # Plot all ROC curves
    plt.figure(figsize=(10,7))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f}, eer = {1:0.4f})'
                   ''.format(roc_auc["micro"], eer["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f}, eer = {1:0.4f})'
                   ''.format(roc_auc["macro"], eer["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f}, eer = {2:0.4f})'
                 ''.format(i, roc_auc[i], eer[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
