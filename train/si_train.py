import numpy as np
import torch
import torch.nn.functional as F

from .train_utils import print_eval
from ..eval.score_utils import embeds_utterance
from sklearn.metrics import roc_curve

def train(config, train_loader, model, optimizer, criterion):
    model.train()
    loss_sum = 0
    accs = []

    print_steps = (np.arange(1,21) * 0.1
                    * len(train_loader)).astype(np.int64)

    splice_frames = config['splice_frames']
    if len(splice_frames) > 1:
        splice_frames_ = np.random.randint(splice_frames[0], splice_frames[1])
    else:
        splice_frames_ = splice_frames[-1]

    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.narrow(2, 0, splice_frames_)

        if not config["no_cuda"]:
            X = X.cuda()
            y = y.cuda()

        optimizer.zero_grad()

        scores = model(X)

        loss = criterion(scores, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        # schedule over iteration
        accs.append(print_eval("train step #{}".format('0'), scores, y,
            loss_sum/(batch_idx+1), display=False))

        del scores
        del loss
        if batch_idx in print_steps:
            print("train, loss: {:.4f}, acc: {:.5f} ".format(loss_sum/(batch_idx+1), np.mean(accs)))

    avg_acc = np.mean(accs)

    return loss_sum, avg_acc

def val(config, val_loader, model, criterion):
    with torch.no_grad():
        model.eval()
        accs = []
        loss_sum = 0
        for (X, y) in val_loader:
            if not config["no_cuda"]:
                X = X.cuda()
                y = y.cuda()
            scores = model(X)
            loss = criterion(scores, y)
            loss_sum += loss.item()
            accs.append(print_eval("dev", scores, y,
                loss.item()))
        avg_acc = np.mean(accs)

        return loss_sum, avg_acc

def sv_test(config, sv_loader, model, trial):
        if isinstance(model, torch.nn.DataParallel):
            model_t = model.module
        else:
            model_t = model

        embeddings, _ = embeds_utterance(config, sv_loader, model_t, None)
        sim_matrix = F.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        cord = [trial.enrolment_id.tolist(), trial.test_id.tolist()]
        score_vector = sim_matrix[cord].numpy()
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
