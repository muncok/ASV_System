import numpy as np
from utils import cos_dist_sim

class spk_model():
    def __init__(self, config, name, enr_keys, enroll_utters):
        self.name = name
        self.accept_thres = config['accept_thres']
        self.enroll_thres = config['enroll_thres']
        self.sim = config['sim']
        self.sv_mode = config['sv_mode']
        self.accept_thres_update = config['accept_thres_update']
        self.enroll_thres_update = config['enroll_thres_update']
        self.n_use_enroll = config['n_use_enroll']
        self.include_init = config['include_init']

        self.n_init_enrolls = len(enroll_utters)
        self.n_total_enroll = self.n_init_enrolls

        self.embed_key = enr_keys
        self.utters = list(enroll_utters)
        self.confidences = [1.0]*self.n_init_enrolls
        self.confidences_scale = [1.0]*self.n_init_enrolls

        self.embed_mean = self._compute_embed_mean()
        self.config = config
        self.cfids = []


    def _compute_embed_mean(self):
        mean_ = np.mean(self.utters, axis=0)

        return mean_

    def enroll(self, key, in_utter, cfid):
        self.embed_key.append(key)
        self.utters.append(in_utter)
        self.confidences.append(cfid)
        self.n_total_enroll += 1
        self.cfids.append(cfid)

        prev_mean = self.embed_mean
        self.embed_mean = self._compute_embed_mean()
        mean_cos = cos_dist_sim(prev_mean, self.embed_mean, dim=0)


        al = self.config['alpha']
        be = self.config['beta']
        if self.config['accept_thres_update']:
            # multiplier
            #self.accept_thres = (self.accept_thres*(1 - al) + al * cfid) * (1 + be * change) # up and down
            #self.accept_thres = self.accept_thres * (1 + al*cfid + be*change) # keep increasing
            #print(al*cfid + be*change)

            # moving average
            T_new = al*cfid*self.accept_thres/self.enroll_thres
            lamb = be*(1-mean_cos)
            self.accept_thres = self.accept_thres*(1-lamb) + T_new*lamb


        if self.config['enroll_thres_update']:
            change = (1-1e-4) - mean_cos
            self.enroll_thres = (self.enroll_thres*(1 - al) + al * cfid ) * (1 + be * change)

        # this value should be not used (it is not allowed supervison)
        if key[:7] == self.name:
            return 1
        else:
            return 0

    def confidence(self, in_utter):
        if self.sim == 'cosMean':
            if self.sv_mode == 'base':
                cfid = cos_dist_sim(self.embed_mean, in_utter, dim=0)
            else:
                ### cosine score ###
                cfid = cos_dist_sim(self.embed_mean, in_utter)
        elif self.sim == 'meanCos':
            in_utter_ = np.array(in_utter).reshape(1,-1)
            if self.sv_mode == 'base':
                utters_ = np.array(self.utters)
            else:
                n_total = len(self.utters)
                if self.n_use_enroll == 'full':
                    idx = list(range(n_total))
                elif self.include_init:
                    idx = [0]+list(range(max(0, n_total-int(self.n_use_enroll))+1, n_total))
                else:
                    idx = list(range(max(0, n_total-int(self.n_use_enroll)), n_total))
                utters_ = np.array(self.utters)[idx]
            cfid = cos_dist_sim(utters_, in_utter_, dim=1).mean()
        elif self.sim == 'euc':
            cfid = 1/(1+np.linalg.norm(self.embed_mean-in_utter, axis=0))

        return cfid

    def show_enrolls(self,):
        print("{:25}: cfid  s_cfid".format("key"))
        for i, (key, cfid, cfid_scale) in enumerate(zip(self.embed_key, self.confidences, self.confidences_scale)):
            if i == self.n_init_enrolls:
                print("==================================")
            print("{}: {:.3f} {:.3f}".format(key, cfid, cfid_scale))

