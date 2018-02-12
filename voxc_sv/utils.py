
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_curve, auc

def roc_auc_eer(dists, labels):
    """
        dists: 1D-Array, [samples]
        labels: 1D-Array, [samples]
    """
    fpr, tpr, thres = roc_curve(labels, dists, pos_label=1)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_ths = thres[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    roc_bundle = {'fpr':fpr, 'tpr':tpr, 'thres':thres}
    np.save(open('roc_data.npy', 'wb'), roc_bundle)

    return roc_auc, eer, eer_ths

def to_np(x):
    return x.data.cpu().numpy()

def spk_verify(spk_model, test_in, sent_label=None, who=False):
    best_score = -2
    pred_spk = 'Unknown'

    for spk in spk_model.keys():
        if sent_label is not None:
            signature = spk_model[spk][sent_label]
            signature_uni = np.mean(spk_model[spk],0)
            score = max(1-cosine(test_in, signature), 1-cosine(test_in,signature_uni))
        else:
            signature_uni = spk_model[spk]
            score = 1-cosine(test_in, signature_uni)

        if score > best_score:
            best_score = score
            pred_spk = spk
    if who:
        return pred_spk, best_score
    else:
        return best_score
