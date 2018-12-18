import numpy as np

class sv_system():
    def __init__(self, spk_models, config):
        self.spk_models = spk_models
        self.sv_mode = config['sv_mode']

        self.log_trial_keys = []
        self.log_trial_cfid = []
        self.log_trial_res  = []
        self.log_trial_mod  = []

    def _get_max_cfid(self, in_utter):
        cfids = []
        for model in self.spk_models:
            cfid = model.confidence(in_utter)
            cfids.append(cfid)

        return np.max(cfids), np.argmax(cfids)

    def verify_and_enroll(self, key, in_utter):
        accept = 0
        enroll = -1
        max_cfid, max_idx = self._get_max_cfid(in_utter)
        spk_model = self.spk_models[max_idx]
        accept_thres = spk_model.accept_thres
        enroll_thres = spk_model.enroll_thres

        if max_cfid > accept_thres:
            accept = 1
        if (max_cfid > enroll_thres) and (self.sv_mode != 'base'):
            enroll = spk_model.enroll(key, in_utter, max_cfid)

        return accept, enroll, max_cfid

    def show_enrolls(self,):
        for model in self.spk_models:
            print(model.name)
            model.show_enrolls()

