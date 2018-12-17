import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve
from tqdm import tqdm

def lda_on_tensor(tensor, lda):
    return torch.from_numpy(lda.transform(tensor.numpy()).astype(np.float32))


def compute_eer(pos_scores, neg_scores, verbose=False):
    score_vector = np.concatenate([pos_scores, neg_scores])
    label_vector = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, thres = roc_curve(label_vector, score_vector, pos_label=1)
    eer = np.min([fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))],
                 1-tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]])
    thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

    if verbose:
        print("eer:{:.3f}% at threshold {:.4f}".format(eer*100, thres))

    return eer, thres

def embeds_utterance(config, val_dataloader, model, lda=None):
    val_iter = iter(val_dataloader)
    embeddings = []
    labels = []
    model.eval()
    if isinstance(config['splice_frames'], list):
        splice_frames = config['splice_frames'][-1]
    else:
        splice_frames = config['splice_frames']

    stride_frames = config['stride_frames']
    # print("spFr:{}, stFr:{}".format(
    with torch.no_grad():
        for batch in tqdm(val_iter):
            x, y = batch
            if not config['no_cuda']:
                x = x.cuda()
            model_outputs = []
            if config['score_mode'] == "precise":
                for i in range(len(x)):
                    out_ = model.embed(x).cpu().detach().data
                    model_outputs.append(out_)
                model_output = torch.cat(model_outputs, dim=0)
            else:
                time_dim = x.size(2)
                split_points = range(0, time_dim-(splice_frames)+1, stride_frames)
                for point in split_points:
                    x_in = x.narrow(2, point, splice_frames)
                    model_outputs.append(model.embed(x_in).detach().cpu().data)
                model_output = torch.stack(model_outputs, dim=0)
                model_output = model_output.mean(0)

            if lda is not None:
                model_output = torch.from_numpy(
                        lda.transform(model_output.numpy()).astype(np.float32))
            embeddings.append(model_output)
            labels.append(y.numpy())
        embeddings = torch.cat(embeddings)
        labels = np.hstack(labels)
    return embeddings, labels

def sv_test(config, sv_loader, model, trial):
        if isinstance(model, torch.nn.DataParallel):
            model_t = model.module
        else:
            model_t = model

        embeddings, _ = embeds_utterance(config, sv_loader, model_t, None)
        score_vector = F.cosine_similarity(
                embeddings[trial.enrolment_id], embeddings[trial.test_id], dim=1)
        label_vector = np.array(trial.label)
        fpr, tpr, thres = roc_curve(
                label_vector, score_vector, pos_label=1)
        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        thres = thres[np.nanargmin(np.abs(fpr - (1 - tpr)))]

        return eer, label_vector, score_vector
