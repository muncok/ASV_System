import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

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
    # plt.figure(figsize=(10,7))
    # plt.plot(fpr["micro"], tpr["micro"],
	 #     label='micro-average ROC curve (area = {0:0.2f}, eer = {1:0.4f})'
		#    ''.format(roc_auc["micro"], eer["micro"]),
	 #     color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
	 #     label='macro-average ROC curve (area = {0:0.2f}, eer = {1:0.4f})'
		#    ''.format(roc_auc["macro"], eer["macro"]),
	 #     color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(n_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
		#  label='ROC curve of {0} (area = {1:0.2f}, eer = {2:0.4f})'
		#  ''.format(i, roc_auc[i], eer[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.show()
