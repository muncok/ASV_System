# coding=utf-8
import numpy as np
import scipy.stats as st
from tqdm import tqdm, tqdm_notebook
from time import sleep

import torch
from torch.autograd import Variable
import torch.utils.data as data

from .train.verification_loss import verification_filter_score as opt_score
from .data import manage_audio
from .data import dataset as dset
from .data import dataloader as dloader

### protonet systems ###

def recToEER(eer_record, thresh_record, verbose=False):
    mean_eer = np.mean(eer_record)
    if verbose:
        lb, ub = st.t.interval(0.95, len(eer_record) - 1, loc=mean_eer, scale=st.sem(eer_record))
        # mean_thresh = np.mean(thresh_record)
        # lb_th, ub_th = st.t.interval(0.95, len(thresh_record) - 1, loc=mean_thresh, scale=st.sem(thresh_record))
        return (mean_eer, ub-mean_eer)
    return mean_eer

def sv_score_sep(opt, val_dataloader, model):
    eer_records = []
    thresh_records = []
    val_iter = iter(val_dataloader)
    nb_splicing = opt.input_length // opt.splice_length
    n_support = opt.num_support_val
    n_classes = opt.classes_per_it_val
    n_query = opt.num_query_val
    for batch in tqdm(val_iter):
        x, y = batch

        postargs = y[:n_classes*(n_support + n_query)]
        classes = np.unique(postargs)
        def supp_idxs(c):
            return torch.LongTensor(np.where(y.numpy() == c)[0][:n_support])
        os_idxs = list(map(supp_idxs, classes))
        oq_idxs = map(lambda c: np.where(y.numpy() == c)
                                [0][n_support:], classes)
        os_x =  x[torch.stack(os_idxs).view(-1)]
        posq_x = x[np.array(list(oq_idxs)).flatten(),]
        negq_x = x[n_classes*(n_support+n_query):,]

        sup_time_dim = x.size(2)
        split_points = range(0, sup_time_dim-(sup_time_dim)//nb_splicing+1, sup_time_dim//nb_splicing)
        os_outputs = []
        for point in split_points:
            os_in = Variable(os_x.narrow(2, point, sup_time_dim//nb_splicing))
            if opt.cuda:
                os_in = os_in.cuda()
            os_outputs.append(model(os_in))
        sup_out = torch.stack(os_outputs, dim=0).mean(0)

        q_time_dim = x.size(2)//2
        split_points = range(0, q_time_dim-(q_time_dim)//nb_splicing+1, q_time_dim//nb_splicing)
        posq_outputs = []
        negq_outputs = []
        for point in split_points:
            posq_in = Variable(posq_x.narrow(2, point, q_time_dim//nb_splicing))
            negq_in = Variable(negq_x.narrow(2, point, q_time_dim//nb_splicing))
            if opt.cuda:
                posq_in = posq_in.cuda()
                negq_in = negq_in.cuda()
            posq_outputs.append(model(posq_in))
            negq_outputs.append(model(negq_in))
        posq_out = torch.stack(posq_outputs, dim=0).mean(0)
        negq_out = torch.stack(negq_outputs, dim=0).mean(0)

        y = Variable(y)
        if opt.cuda:
            y = y.cuda()
        from verification_loss import verification_sep_score
        eer, thresh = verification_sep_score(sup_out, posq_out, negq_out, classes)
        eer_records.append(eer)
        thresh_records.append(thresh)
    sleep(0.05)
    mean_eer = np.mean(eer_records)
    mean_thresh = np.mean(thresh_records)
    lb, ub = st.t.interval(0.95, len(eer_records)-1, loc=mean_eer, scale=st.sem(eer_records))
    lb_th, ub_th = st.t.interval(0.95, len(thresh_records)-1, loc=mean_thresh, scale=st.sem(thresh_records))
    print("eer: {:.2f}% +- {:.2f}%, thresh: {:.5f} +- {:.5f}".format(mean_eer*100, (ub-mean_eer)*100,
                                                                     mean_thresh, (ub_th-mean_thresh)))

def sv_score(opt, val_dataloader, model, filter_types, lda=None):
    if opt.cuda:
        model = model.cuda()
    val_iter = iter(val_dataloader)
    # nb_splicing = opt.input_length // opt.splice_length
    eer_records = {k:[] for k in filter_types}
    thresh_records = {k:[] for k in filter_types}
    lda_eer_records = {k:[] for k in filter_types}
    lda_thresh_records = {k:[] for k in filter_types}
    model.eval()
    for batch in tqdm_notebook(val_iter):
        x, y = batch
        model_outputs = []
        time_dim = x.size(2)
        splice_dim = opt.splice_length
        split_points = range(0, time_dim-splice_dim, splice_dim)
        for point in split_points:
            x_in = Variable(x.narrow(2, point, splice_dim))
            if opt.cuda:
                x_in = x_in.cuda()
            embed = model.embed(x_in)
            model_outputs.append(embed)
        model_output = torch.stack(model_outputs, dim=-1)
        if lda:
            lda_output = model_output.cpu().data.numpy()
            s, d, k = lda_output.shape
            lda_output = np.transpose(lda_output, [0,2,1]).reshape(-1, d)
            lda_output = lda.transform(lda_output).astype(np.float32)
            lda_output = lda_output.reshape(s,k,-1).transpose([0,2,1])
            if opt.cuda:
                lda_output = Variable(torch.from_numpy(lda_output).cuda())
        y = Variable(y)
        if opt.cuda:
            y = y.cuda()

        for filter_type in filter_types:
            eer, thresh = opt_score(model_output, target=y, n_classes=opt.classes_per_it_val,
                                    n_support=opt.num_support_val, n_query=opt.num_query_val,
                                    n_frames=opt.num_test_frames, filter_type=filter_type)
            eer_records[filter_type].append(eer)
            thresh_records[filter_type].append(thresh)
            if lda:
                eer, thresh = opt_score(lda_output, target=y, n_classes=opt.classes_per_it_val,
                                        n_support=opt.num_support_val, n_query=opt.num_query_val,
                                        n_frames=opt.num_test_frames, filter_type=filter_type)
                lda_eer_records[filter_type].append(eer)
                lda_thresh_records[filter_type].append(thresh)

    return_val = dict()
    for filter_type in filter_types:
        eer_record = eer_records[filter_type]
        thresh_record = thresh_records[filter_type]
        return_val[filter_type] = recToEER(eer_record, thresh_record, verbose=True)
    if lda:
        for filter_type in filter_types:
                eer_record = lda_eer_records[filter_type]
                thresh_record = lda_thresh_records[filter_type]
                return_val[filter_type+"_lda"] = recToEER(eer_record, thresh_record, verbose=True)
    return return_val

def posterior_prob(opt, val_dataloader, model):
    val_iter = iter(val_dataloader)
    for batch in tqdm(val_iter):
        x, y = batch
        x_in = Variable(x)
        if opt.cuda:
            x_in = x_in.cuda()
        model_output = model(x_in)
        _, y_hat = torch.max(model_output, dim=1)

### honk_sv system ###
def embed(config, model, audio_path):
    data = manage_audio.preprocess_from_path(config, audio_path)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    data_in = Variable(data, requires_grad=False).unsqueeze(0)
    if not config["no_cuda"]:
        data_in = data_in.cuda()
    feature = model.embed(data_in).cpu().data.numpy()
    return feature

def enroll_uttr(config, model, test_loader=None):
    if not test_loader:
        _, _, test_set = dset.SpeechDataset.read_manifest(config)
        test_loader = data.DataLoader(test_set, batch_size=config['batch_size'])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    embed_size = model.feat_size
    embeds = np.zeros([embed_size])
    for enroll_in, labels in test_loader:
        enroll_in = Variable(enroll_in, requires_grad=False)
        if not config["no_cuda"]:
            enroll_in = enroll_in.cuda()
        feature = model.embed(enroll_in)
        numeric_feature = feature.cpu().data.numpy()
        # accumulates features
        embeds += np.sum(numeric_feature, axis=0)

    # averaging the features for making signatures
    spk_models= embeds / len(test_loader.dataset.audio_labels)
    return spk_models

def enroll_frame(config, model, test_loader=None, _collate_fn=None):
    if not test_loader:
        datasets = dset.SpeechDataset.read_manifest(config)
        _,_,test_loader = dloader.get_loader(config, datasets, _collate_fn)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    embed_size = model.feat_size
    embeds = np.zeros([embed_size])
    counts = 0
    for enroll_total, labels in test_loader:
        counts += len(enroll_total)
        for i in range(0,len(enroll_total), 64):
            enroll_in = Variable(enroll_total[i:i+64], requires_grad=False)
            if not config["no_cuda"]:
                enroll_in = enroll_in.cuda()
            feature = model.embed(enroll_in)
            numeric_feature = feature.cpu().data.numpy()
            # accumulates features
            embeds += np.sum(numeric_feature, axis=0)
    # averaging the features for making signatures
    spk_models= embeds / counts
    return spk_models

def dvector(config, model, audio_path):
    data = manage_audio.preprocess_from_path(config, audio_path)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    uttr_data = []
    half_splice = config['splice_length'] // 2
    points = np.arange(half_splice, len(data) - half_splice)
    for point in points:
        uttr_data.append(data[point-half_splice:point+half_splice])
    uttr_data = torch.stack(uttr_data)

    feature = np.zeros((model.feat_size,))
    for i in range(0,len(uttr_data)//1, 64):
        frame_in = Variable(uttr_data[i:i+64], requires_grad=False)
        if not config["no_cuda"]:
            frame_in = frame_in.cuda()
        feature +=  np.sum(model.embed(frame_in).cpu().data.numpy(), 0)
    feature = feature / (len(uttr_data)//1)
    return feature

