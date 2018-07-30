from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F

from train.train_utils import print_eval
from sv_score.score_utils import embeds_utterance
from sklearn.metrics import roc_curve

def train(config, train_loader, model, optimizer, criterion):
    model.train()
    loss_sum = 0
    accs = []

    print_steps = np.array([0.25, 0.5, 0.75, 0.99]) * len(train_loader)
    for batch_idx, (X, y) in tqdm(enumerate(train_loader), ncols=100,
            total=len(train_loader)):
        # X_batch = (batch, channel, time, bank)
        input_frames = np.random.randint(300, 800)
        start_frame = np.random.randint(0, 800-input_frames)
        X = X.narrow(2, start_frame, input_frames)
        if not config["no_cuda"]:
            X = X.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        scores = model(X)
        loss = criterion(scores, y)
        loss_sum += loss.item()
        loss.backward()
        # learning rate change
        optimizer.step()
        # schedule over iteration
        accs.append(print_eval("train step #{}".format('0'), scores, y,
            loss_sum/(batch_idx+1), display=False))
        if batch_idx in print_steps:
            print("train acc: {}/{}".format(batch_idx, len(train_loader)))

    avg_acc = np.mean(accs)

    return loss_sum, avg_acc

def val(config, val_loader, model, criterion):
    with torch.no_grad():
        model.eval()
        accs = []
        loss_sum = 0
        for (X, y) in tqdm(val_loader, ncols=100,
                total=len(val_loader)):
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

        embeddings, _ = embeds_utterance(config, sv_loader,
                model_t, None)
        sim_matrix = F.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        cord = [trial.enrolment_id.tolist(), trial.test_id.tolist()]
        score_vector = sim_matrix[cord].numpy()
        label_vector = np.array(trial.label)
        fpr, tpr, thres = roc_curve(
                label_vector, score_vector, pos_label=1)
        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]

        return eer, label_vector, score_vector
