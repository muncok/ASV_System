import os
import pickle
import pandas as pd
import numpy as np
import argparse

from multiprocessing import Process, Manager
from tqdm import tqdm
from utils import key2df
from utils import set_threshold
from evaluation import eval_wrapper

parser = argparse.ArgumentParser()

parser.add_argument('-n_enr',
                    type=str,
                    help='number of enrollments',
                    default='full')

parser.add_argument('-out_dir',
                    type=str,
                    help='output_dir base',
                    required=True
                    )

parser.add_argument('-n_process',
                    type=int,
                    help='number of processes',
                    default=40)

parser.add_argument('-sv_mode',
                    type=str,
                    help='type of sv_system',
                    choices=['base', 'inc'],
                    required=True
                    )

parser.add_argument('-trial_t',
                    type=str,
                    help='type of trial',
                    choices=['random', 'sortedPos'],
                    required=True
                    )

parser.add_argument('-ths_t',
                    type=str,
                    help='type of threshold',
                    choices=['normal', 'extreme'],
                    default='normal'
                    )

parser.add_argument('-update',
                    help='use of thresh update',
                    action='store_true')

parser.add_argument('-incl_init',
                    help='include the init enrollment',
                    action='store_true')

if __name__=='__main__':

    args = parser.parse_args()

    config = {
            # sim: cosMean, meanCos, euc
            'sim': 'meanCos',
            'accept_thres_update': args.update,
            'enroll_thres_update': False,
            # trial_tpye: sortedPos, random, posNeg
            'trial_type': args.trial_t,
            # n_use_enroll: 'full' or 'N' (decimal in string type such as '5')
            'n_use_enroll': args.n_enr,
            'include_init': args.incl_init,
            'alpha': 0.0005,  #alpha 0.0005
            'beta': 0.01,  #beta 0.01
            # normal or extreme threshold values
            'thresh_type': args.ths_t,
            'sv_mode': args.sv_mode
            }
    print(config)

    keys = np.array(pickle.load(open("xvector_embeds/sv_keys.pkl", "rb")))
    embeds = np.load("xvector_embeds/sv_embeds.npy")
    key_df = key2df(keys)
    key2id = {k:v for v, k in enumerate(keys)}

    # trial for finding best threshold
    trial_base = "trials/enr306_uttr1/"
    set_threshold(config, embeds, trial_base+'trial_for_thresh.pkl')
    print('Accept Thres: {:.5f}, Enroll Thres: {:.5f}'.format(
        config['accept_thres'], config['enroll_thres']))

# ===================== Run trials =====================
    n_parallel = args.n_process
    #for p_ratio in [0.01, 0.1, 0.5, 0.9]:
    for p_ratio in [0.9]:
        print("="*100)
        print("p_ratio: {}".format(p_ratio))
        trial_set = pickle.load(open(trial_base+"trials_ratio_{}.pkl".format(str(p_ratio)), "rb"))
        # ===== Output Dir =====
        output_dir = "{}/n_enr_{}_pRatio_{}".format(
                args.out_dir, config['n_use_enroll'], p_ratio)
        if config["include_init"]:
            output_dir += "_initEnr"
        if config["accept_thres_update"]:
            output_dir += "_thresUpdt"
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        pickle.dump(config, open(output_dir+'/config.pkl', "wb"))

        metaR_l = []
        pScore_l = []
        nScore_l = []
        for i, idx in enumerate(range(0,len(trial_set), n_parallel)):
            procs = []
            manager = Manager()
            metaR_q = manager.Queue()
            pScore_q = manager.Queue()
            nScore_q = manager.Queue()

            print('Starting jobs [{}/{}]'.format(i, len(trial_set)//n_parallel))
            for j, trial in enumerate(trial_set[idx:idx+n_parallel]):
                enr_spks, enr_uttr_keys, pos_trial_keys, neg_trial_keys = trial
                n_trials = len(pos_trial_keys) + len(neg_trial_keys)
                enr_id = np.array([key2id[k] for k in enr_uttr_keys])
                if config['trial_type'] == 'random':
                    permu_idx = np.random.permutation(range(n_trials))
                    trials_id = np.array([key2id[k] for k in pos_trial_keys + neg_trial_keys])[permu_idx]
                    label = np.array([1]*len(pos_trial_keys) + [0]*len(neg_trial_keys))
                    label = label[permu_idx]
                elif config['trial_type'] == 'sortedPos':
                    sessions = list(map(lambda x: x[8:19], pos_trial_keys))
                    df = pd.DataFrame.from_dict(dict( utters = pos_trial_keys, session = sessions ))
                    unique_session = np.unique(sorted(df.session.values))
                    session_cnt = df.session.value_counts()

                    n_pos_trials = len(pos_trial_keys)
                    n_unique_sess = len(unique_session)
                    n_sess_trials = len(neg_trial_keys)+n_unique_sess

                    pos_sess_idx_ = sorted(np.random.choice(range(n_sess_trials), size=n_unique_sess, replace=False))

                    pos_seat_idx_ = []
                    for i, sess in enumerate(unique_session):
                        l = session_cnt[sess]
                        pos_sess_idx_[i+1:] += l-1
                        for j in range(l):
                            pos_seat_idx_.append(j+pos_sess_idx_[i])

                    neg_seat_idx_ = list(set(range(n_trials)) - set(pos_seat_idx_))

                    pos_trial_id = [key2id[k] for k in sorted(pos_trial_keys)]
                    neg_trial_id = [key2id[k] for k in neg_trial_keys]
                    trials_id = np.zeros(n_trials)
                    trials_id[pos_seat_idx_] = pos_trial_id
                    trials_id[neg_seat_idx_] = neg_trial_id
                    trials_id = trials_id.astype(np.int64)

                    label = np.zeros(n_trials)
                    label[pos_seat_idx_] = [1]*len(pos_trial_keys)
                    label[neg_seat_idx_] = [0]*len(neg_trial_keys)

                proc = Process(target=eval_wrapper,
                        args=(config, embeds, keys, enr_spks, enr_id, trials_id, label,
                            metaR_q, pScore_q, nScore_q))
                procs.append(proc)
                proc.start()

            print('Joining jobs [{}/{}]'.format(i, len(trial_set)//n_parallel))
            for p in tqdm(procs):
                metaR_l.append(metaR_q.get())
                pScore_l += pScore_q.get()
                nScore_l += nScore_q.get()
                p.join()

        pickle.dump(metaR_l, open("{}/result.pkl".format(output_dir), "wb"))
        pickle.dump(pScore_l, open("{}/posScores.pkl".format(output_dir), "wb"))
        pickle.dump(nScore_l, open("{}/negScores.pkl".format(output_dir), "wb"))

    print('Done')
