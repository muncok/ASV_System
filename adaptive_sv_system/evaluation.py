import numpy as np
from spk_model import spk_model
from sv_system import sv_system
from utils import cos_dist_sim, compute_eer

def run_inc_sv_system(sv_system, embeds, keys, trial_ids, label):
    # run incremental speaker verification system
    accepts = []
    is_enrolls = []
    scores = []
    for trial_id in trial_ids:
        # evaluate trial samples step by step
        accept, enroll, cfid = sv_system.verify_and_enroll(keys[trial_id],
                embeds[trial_id])
        accepts.append(accept)
        is_enrolls.append(enroll)
        scores.append(cfid)

    ### Accuracy ###
    acc = np.count_nonzero(np.array(accepts) == label) / len(label)
    correct = np.count_nonzero(np.array(accepts) == label)
    wrong = len(label) - correct

    ### Enroll Accuracy ###
    n_total_enrolls = np.count_nonzero(np.array(is_enrolls) != -1)
    if n_total_enrolls == 0:
        enr_acc = 1
    else:
        enr_acc = np.count_nonzero(np.array(is_enrolls) == 1) / n_total_enrolls

    ### FPR and FNR
    fpr = np.count_nonzero((np.array(accepts) == 1) & (label == 0)) /\
            np.count_nonzero(label == 0)
    fnr = np.count_nonzero((np.array(accepts) == 0) & (label == 1)) /\
            np.count_nonzero(label == 1)

    return acc, enr_acc, fpr, fnr, is_enrolls, scores, correct, wrong

def evaluation_base(config, embeds, enr_spk, enr_id, trials_id, label):
    #enr_uttrs_embeds = embeds[[key2id[k] for k in enr_uttrs]]
    enr_uttrs_embeds = embeds[enr_id]
    trial_uttrs_embeds = embeds[trials_id]
    scores = cos_dist_sim(enr_uttrs_embeds, trial_uttrs_embeds, dim=1)
    scores = scores.mean(0).astype(np.float32)
    accepts = scores > config['accept_thres']
    acc = np.count_nonzero(np.array(accepts) == label) / len(label)
    n_correct = np.count_nonzero(np.array(accepts) == label)
    n_wrong = len(label) - n_correct
    fpr = np.count_nonzero((np.array(accepts) == True) & (label == 0)) /\
            np.count_nonzero(label == 0)
    fnr = np.count_nonzero((np.array(accepts) == False) & (label == 1)) /\
            np.count_nonzero(label == 1)
    pos_scores = scores[label==1]
    neg_scores = scores[label==0]
    eer = compute_eer(pos_scores, neg_scores)
    pScore = pos_scores.tolist()
    nScore = neg_scores.tolist()

    return [n_correct, n_wrong, acc, eer, fpr, fnr],  pScore, nScore


def evaluation_inc(config, embeds, keys, enr_spk, enr_id, trials_id, label):
    spk_models = []
    enroll_utters = embeds[enr_id]
    enr_keys = [keys[id_] for id_ in enr_id]
    spk_models.append(spk_model(config, enr_spk, enr_keys, enroll_utters))

    system = sv_system(spk_models, config)
    acc, enr_acc, fpr, fnr, is_enrolls, scores, n_correct, n_wrong =\
            run_inc_sv_system(system, embeds, keys, trials_id, label)
    scores = np.array(scores)
    pos_scores = scores[label==1]
    neg_scores = scores[label==0]
    eer = compute_eer(pos_scores, neg_scores)
    pScore = pos_scores.tolist()
    nScore = neg_scores.tolist()

    return [n_correct, n_wrong, acc, eer, fpr, fnr, is_enrolls, enr_acc], pScore, nScore

def eval_wrapper(config, embeds, keys, enr_spk, enr_id, trials_id, label,
        metaR_q, pScore_q, nScore_q):
    n_trials = len(trials_id)
    if config['sv_mode'] == 'base':
        base_stat = evaluation_base(config, embeds, enr_spk, enr_id, trials_id, label)
        metaR, pos_scores, neg_scores = base_stat
        result = [enr_spk, enr_id[0], n_trials] + metaR
    elif config['sv_mode'] == 'inc':
        inc_stat = evaluation_inc(config, embeds, keys, enr_spk, enr_id, trials_id, label)
        metaR,  pos_scores, neg_scores = inc_stat
        result = [enr_spk, enr_id[0], n_trials] + metaR
    else:
        raise NotImplemented

    metaR_q.put(result)
    pScore_q.put(pos_scores)
    nScore_q.put(neg_scores)
