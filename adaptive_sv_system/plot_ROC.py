from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
#import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

# def plot_ROC(y_train_true, y_train_prob, y_test_true, y_test_prob):
def plot_ROC(y_train_true, y_train_prob):
    '''
    a funciton to plot the ROC curve for train labels and test labels.
    Use the best threshold found in train set to classify items in test set.
    '''
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train_true, y_train_prob, pos_label =True)
    sum_sensitivity_specificity_train = tpr_train + (1-fpr_train)
    best_threshold_id_train = np.argmax(sum_sensitivity_specificity_train)
    best_threshold = thresholds_train[best_threshold_id_train]
    best_fpr_train = fpr_train[best_threshold_id_train]
    best_tpr_train = tpr_train[best_threshold_id_train]
    y_train = y_train_prob > best_threshold

    cm_train = confusion_matrix(y_train_true, y_train)
    acc_train = accuracy_score(y_train_true, y_train)
    auc_train = roc_auc_score(y_train_true, y_train)

    print ('Train Accuracy: %s ' %acc_train)
#     print ('Train AUC: %s ' %auc_train)
#     print ('Train Confusion Matrix:')
#     print (cm_train)

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(121)
    # curve1 = ax.plot(fpr_train, tpr_train)
    # curve2 = ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # dot = ax.plot(best_fpr_train, best_tpr_train, marker='o', color='black')
    # ax.text(best_fpr_train, best_tpr_train, s = '(%.3f,%.3f)' %(best_fpr_train, best_tpr_train))
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve (Train), AUC = %.4f'%auc_train)
    # plt.show()

    return best_threshold, fpr_train, thresholds_train
