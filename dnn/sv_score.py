# coding=utf-8
import numpy as np
import scipy.stats as st
from tqdm import tqdm

import torch
from torch.autograd import Variable

# from .train.verification_loss import verification_eer_score as eer_score

def recToEER(eer_record, thresh_record, verbose=False):
    mean_eer = np.mean(eer_record)
    if verbose:
        lb, ub = st.t.interval(0.95, len(eer_record) - 1, loc=mean_eer, scale=st.sem(eer_record))
        # mean_thresh = np.mean(thresh_record)
        # lb_th, ub_th = st.t.interval(0.95, len(thresh_record) - 1, loc=mean_thresh, scale=st.sem(thresh_record))
        return (mean_eer, ub-mean_eer)
    return mean_eer

def embeds(opt, val_dataloader, model, lda=None):
    val_iter = iter(val_dataloader)
    model.eval()
    splice_dim = opt.splice_frames
    embeddings = []
    labels = []
    if lda is not None:
        print("LDA is loaded")
    for batch in (val_iter):
        x, y = batch
        print(x.shape)
        time_dim = x.size(2)
        split_points = range(0, time_dim-splice_dim+1, splice_dim)
        model_outputs = []
        for point in split_points:
            x_in = Variable(x.narrow(2, point, splice_dim))
            if opt.cuda:
                x_in = x_in.cuda()
            model_outputs.append(model.embed(x_in).cpu().data)
        model_output = torch.stack(model_outputs, dim=0)
        model_output = model_output.mean(0)
        if lda is not None:
            model_output = torch.from_numpy(lda.transform(model_output.numpy()).astype(np.float32))
        embeddings.append(model_output)
        labels.append(y.numpy())
    embeddings = torch.cat(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels

# def sv_score(opt, val_dataloader, model, filter_types, lda=None):
#     """
#     sv scoring using verification_batch_sampler.
#     :param opt:
#     :param val_dataloader: dataloader with verification_batch_sampler
#     :param model:
#     :param filter_types:
#     :param lda:
#     :return:
#     """
#     if opt.cuda:
#         model = model.cuda()
#
#     val_iter = iter(val_dataloader)
#     eer_records = {k:[] for k in filter_types}
#     thresh_records = {k:[] for k in filter_types}
#     lda_eer_records = {k:[] for k in filter_types}
#     lda_thresh_records = {k:[] for k in filter_types}
#
#     model.eval()
#     for batch in tqdm(val_iter):
#         x, y = batch
#         model_outputs = []
#         time_dim = x.size(2)
#         splice_frames = opt.splice_length
#         split_points = range(0, time_dim-splice_frames, splice_frames)
#         for point in split_points:
#             x_in = Variable(x.narrow(2, point, splice_frames))
#             if opt.cuda:
#                 x_in = x_in.cuda()
#             embed = model.embed(x_in)
#             model_outputs.append(embed)
#         model_output = torch.stack(model_outputs, dim=-1)
#         if lda is not None:
#             lda_output = model_output.cpu().data.numpy()
#             s, d, k = lda_output.shape
#             lda_output = np.transpose(lda_output, [0,2,1]).reshape(-1, d)
#             lda_output = lda.transform(lda_output).astype(np.float32)
#             lda_output = lda_output.reshape(s,k,-1).transpose([0,2,1])
#             if opt.cuda:
#                 lda_output = Variable(torch.from_numpy(lda_output).cuda())
#         y = Variable(y)
#         if opt.cuda:
#             y = y.cuda()
#
#         for filter_type in filter_types:
#             eer, thresh = eer_score(model_output, target=y, n_classes=opt.classes_per_it_val,
#                                     n_support=opt.num_support_val, n_query=opt.num_query_val,
#                                     n_frames=opt.num_test_frames, filter_type=filter_type)
#             eer_records[filter_type].append(eer)
#             thresh_records[filter_type].append(thresh)
#             if lda:
#                 eer, thresh = eer_score(lda_output, target=y, n_classes=opt.classes_per_it_val,
#                                         n_support=opt.num_support_val, n_query=opt.num_query_val,
#                                         n_frames=opt.num_test_frames, filter_type=filter_type)
#                 lda_eer_records[filter_type].append(eer)
#                 lda_thresh_records[filter_type].append(thresh)
#
#     return_val = dict()
#     for filter_type in filter_types:
#         eer_record = eer_records[filter_type]
#         thresh_record = thresh_records[filter_type]
#         return_val[filter_type] = recToEER(eer_record, thresh_record, verbose=True)
#     if lda:
#         for filter_type in filter_types:
#                 eer_record = lda_eer_records[filter_type]
#                 thresh_record = lda_thresh_records[filter_type]
#                 return_val[filter_type+"_lda"] = recToEER(eer_record, thresh_record, verbose=True)
#     return return_val

