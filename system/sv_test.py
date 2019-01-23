import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve
from tqdm import tqdm
from .compute_min_dcf import ComputeErrorRates, ComputeMinDcf

def extract_embed_var_len(config, val_dataloader, model):
    # each input has different length
    # keep their own length
    val_iter = iter(val_dataloader)
    embeddings = []
    labels = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_iter):
            seq_len, x, y = batch

            if not config['no_cuda']:
                x = x.cuda()

            model_outputs = []
            for i in range(len(x)):
                x_in = x[i:i+1,:,:seq_len[i]]
                out_ = model.embed(x_in).cpu().detach().data
                model_outputs.append(out_)
            model_output = torch.cat(model_outputs, dim=0)
            embeddings.append(model_output)
            labels.append(y.numpy())

    embeddings = torch.cat(embeddings)
    labels = np.hstack(labels)

    return embeddings, labels

def extract_embed_fix_len(config, val_dataloader, model):
    # evary inputs have same length
    # they are chosen by fixed length
    val_iter = iter(val_dataloader)
    embeddings = []
    labels = []
    model.eval()

    if isinstance(config['splice_frames'], list):
        splice_frames = config['splice_frames'][-1]
    else:
        splice_frames = config['splice_frames']
    stride_frames = config['stride_frames']

    with torch.no_grad():
        for batch in tqdm(val_iter):
            x, y = batch

            if not config['no_cuda']:
                x = x.cuda()

            model_outputs = []
            time_dim = x.size(2)
            # input are splitted by amount of splice_frames
            split_points = range(0, time_dim-(splice_frames)+1, stride_frames)
            for point in split_points:
                x_in = x.narrow(2, point, splice_frames)
                model_outputs.append(model.embed(x_in).detach().cpu().data)
            model_output = torch.stack(model_outputs, dim=0)
            # splitted snipets are averaged
            model_output = model_output.mean(0)
            embeddings.append(model_output)
            labels.append(y.numpy())

    embeddings = torch.cat(embeddings)
    labels = np.hstack(labels)

    return embeddings, labels

def get_embeds(config, sv_loader, model):
    if isinstance(model, torch.nn.DataParallel):
        model_t = model.module
    else:
        model_t = model

    if config['input_clip']:
        embeddings, _ = extract_embed_fix_len(config, sv_loader, model_t)
    else:
        embeddings, _ = extract_embed_var_len(config, sv_loader, model_t)

    return embeddings

def compute_minDCF(scores, labels, c_miss=1.0, c_fa=1.0, p_target=0.01):
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target,
         c_miss, c_fa)
    print("minDCF is {0:.4f} at threshold {1:.4f} (p-target={2}, c-miss={3},"
        " c-fa={4})".format(mindcf, threshold, p_target,c_miss, c_fa))

def sv_test(embeddings, trial):
    score_vector = F.cosine_similarity(
            embeddings[trial.enroll_idx], embeddings[trial.test_idx],
            dim=1).numpy().tolist()
    label_vector = np.array(trial.label)
    fpr, tpr, thres = roc_curve(
            label_vector, score_vector, pos_label=1)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    return eer, thres, score_vector

def get_trial_result(trial, scores):
    # result = pd.DataFrame({'enr':trial.enroll_id, 'test':trial.test_id,
        # 'score':scores, 'label':trial.label})
    result = pd.DataFrame({'score':scores})

    return result
