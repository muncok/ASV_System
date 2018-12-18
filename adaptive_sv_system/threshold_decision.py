import numpy as np
import pandas as pd
from plot_ROC import plot_ROC
import torch
from torch.nn.funtional import cosine_similarity

def cos_dist_sim_torch(a, b, dim=0):
    a = torch.from_numpy(a).float()
    b = torch.from_numpy(b).float()

    return cosine_similarity(a, b, dim=dim).numpy()

def euc_dist(a, b, dim):
    return np.linalg.norm(a-b, axis=dim)

def euc_dist_sim(a, b, dim=0):
    return 1/(1+euc_dist(a, b, dim))


def get_threshold(config):
    case = 'enr10_pos300'

    trial_for_thresh = pd.read_pickle('./cases/'+case+'/validation/val_thresh_trials.pkl')


    if ('Cos' in config['sim']) or ('cos' in config['sim']):
        train_score_vector = cos_dist_sim_torch(enroll_embeds[trial_for_thresh.enrolment_id],
                                          enroll_embeds[trial_for_thresh.test_id], dim=1)
    elif 'euc' in config['sim']:
        train_score_vector =  euc_dist_sim(enroll_embeds[trial_for_thresh.enrolment_id],
                                           enroll_embeds[trial_for_thresh.test_id], dim=1)

    train_label_vector = trial_for_thresh.label.tolist()
    accept_thres, fpr_, thres_ = plot_ROC(train_label_vector, train_score_vector)

    config['accept_thres'] = thres_[np.where(fpr_ > 0.2)[0][0]]
    config['enroll_thres'] = thres_[np.where(fpr_ < 0.01)[0][-1]]
    print('Accept Thres: {:.5f}, Enroll Thres: {:.5f}'.format(config['accept_thres'], config['enroll_thres']))
