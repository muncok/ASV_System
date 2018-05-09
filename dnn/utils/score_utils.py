import numpy as np
from sklearn.metrics import roc_curve

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

