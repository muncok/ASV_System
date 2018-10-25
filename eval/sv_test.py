import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve
from tqdm import tqdm
def embeds_utterance(config, val_dataloader, model, lda=None):
    val_iter = iter(val_dataloader)
    embeddings = []
    labels = []
    model.eval()
    if isinstance(config['splice_frames'], list):
        splice_frames = config['splice_frames'][-1]
    else:
        splice_frames = config['splice_frames']
    # splice_frames = config['input_frames']

    stride_frames = config['stride_frames']
    # print("spFr:{}, stFr:{}".format(
    with torch.no_grad():
        for batch in tqdm(val_iter):
            seq_lens, x, y = batch
            if not config['no_cuda']:
                x = x.cuda()
            model_outputs = []
            if config['score_mode'] == "precise":
                for i in range(len(x)):
                    out_ = model.embed(x[i,:,:seq_lens[i]]).cpu().detach().data
                    model_outputs.append(out_)
                model_output = torch.cat(model_outputs, dim=0)
            else:
                time_dim = x.size(2)
                split_points = range(0, time_dim-(splice_frames)+1,
                        stride_frames)
                for point in split_points:
                    x_in = x.narrow(2, point, splice_frames)
                    model_outputs.append(model.embed(x_in).cpu().data)
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

def embeds_utterance_frames(config, splice_frames, val_dataloader, model):
    val_iter = iter(val_dataloader)
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in val_iter:
            x, y = batch
            if not config['no_cuda']:
                x = x.cuda()
            x = x.narrow(2, 0, splice_frames)
            e = model.embed(x)
            embeddings.append(e)
            labels.append(y.numpy())
        embeddings = torch.cat(embeddings)
        labels = np.hstack(labels)
    return embeddings, labels

def embeds_sub_utterance(config, val_dataloader, model, lda=None):
    val_iter = iter(val_dataloader)
    embeddings = []
    labels = []
    model.eval()
    assert len(config['splice_frames']) == 1
    splice_frames = config['splice_frames'][0]
    # splice_frames = config['input_frames']

    stride_frames = config['stride_frames']
    # print("spFr:{}, stFr:{}".format(
    with torch.no_grad():
        for batch in val_iter:
            x, y = batch
            if not config['no_cuda']:
                x = x.cuda()
            if config['score_mode'] == "precise":
                model_output = model.embed(x).cpu().data
            else:
                model_outputs = []
                time_dim = x.size(2)
                split_points = range(0, time_dim-(splice_frames)+1,
                        stride_frames)
                # print(split_points)
                for point in split_points:
                    x_in = x.narrow(2, point, splice_frames)
                    model_outputs.append(model.embed(x_in).cpu().data)
                model_output = torch.stack(model_outputs, dim=1)
                # model_output = model_output.mean(0)

            if lda is not None:
                model_output = torch.from_numpy(
                        lda.transform(model_output.numpy()).astype(np.float32))
            embeddings.append(model_output)
            labels.append(y.numpy())
        embeddings = torch.cat(embeddings, dim=0)
        # print(embeddings.shape)
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

        return eer, thres

def sv_test1(config, sv_loader, model, trial):
        """ differnce length btw enroll and test """
        if isinstance(model, torch.nn.DataParallel):
            model_t = model.module
        else:
            model_t = model

        config['splice_frames'] = config['input_frames']
        enroll_embeddings, _ = embeds_utterance(config, sv_loader, model_t, None)

        config['splice_frames'] = 200
        config['stride_frames'] = config['input_frames'] # for avoiding aggregating
        test_embeddings, _ = embeds_utterance(config, sv_loader, model_t, None)
        score_vector = F.cosine_similarity(
                enroll_embeddings[trial.enrolment_id], test_embeddings[trial.test_id], dim=1)
        label_vector = np.array(trial.label)
        fpr, tpr, thres = roc_curve(
                label_vector, score_vector, pos_label=1)
        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]

        return eer, label_vector, score_vector

def sv_euc_test(config, sv_loader, model, trial):
        if isinstance(model, torch.nn.DataParallel):
            model_t = model.module
        else:
            model_t = model

        embeddings, _ = embeds_utterance(config, sv_loader, model_t, None)
        embeddings /= embeddings.norm(dim=1,keepdim=True)
        a = embeddings.unsqueeze(1)
        b = embeddings.unsqueeze(0)
        dist = a - b
        sim_matrix = -dist.norm(dim=2)
        cord = [trial.enrolment_id.tolist(), trial.test_id.tolist()]
        score_vector = sim_matrix[cord].numpy()
        label_vector = np.array(trial.label)
        fpr, tpr, thres = roc_curve(
                label_vector, score_vector, pos_label=1)
        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]

        return eer, label_vector, score_vector
