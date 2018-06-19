import os
import random
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm_notebook
from tqdm import tqdm

def make_abspath(rel_path):
    if not os.path.isabs(rel_path):
        rel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel_path)
    return rel_path


def print_eval(name, scores, labels, loss, end="\n", verbose=False, binary=False):
    batch_size = labels.size(0)
    if not binary:
        accuracy = (torch.max(scores, 1)[1] == labels.data).sum().float() / batch_size
    else:
        preds = (scores.data > 0.5)
        targets = (labels.data == 1)
        accuracy = (preds == targets).sum() / batch_size
    if verbose:
        tqdm.write("{} accuracy: {:>3}, loss: {:<7}".format(name, accuracy, loss), end=end)
    return accuracy

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def evaluate(config, model, test_loader):
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    splice_frames = config["splice_frames"]
    criterion = nn.CrossEntropyLoss()
    model.eval()
    accs = []
    for X_batch, y_batch in tqdm_notebook(test_loader, total=len(test_loader)):
        timedim = X_batch.size(2)
        for i in range(0, timedim - splice_frames+1 , splice_frames):
            X = X_batch.narrow(2, i, splice_frames)
            y = y_batch
            if not config["no_cuda"]:
                X = X.cuda()
                y = y.cuda()
            scores = model(X)
            loss = criterion(scores, y)
            accs.append(print_eval("test", scores, y, loss.item()))
    avg_acc = np.mean(accs)
    print("final test accuracy: {}".format(avg_acc))


def si_train(config, loaders, model, tqdm_v=tqdm):
    train_loader, dev_loader, test_loader = loaders
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    # optimizer
    learnable_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][0], nesterov=config["use_nesterov"],
                                weight_decay=config["weight_decay"], momentum=config["momentum"])
    criterion = nn.CrossEntropyLoss()

    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    max_acc = 0
    step_no = 0
    splice_frames = config["splice_frames"]
    stride_frames = config["stride_frames"]
    print_step = config["print_step"]
    # training iteration
    for epoch_idx in range(config["n_epochs"]):
        for batch_idx, (X_batch, y_batch) in tqdm_v(enumerate(train_loader), total=len(train_loader)):
            # X_batch = (batch, channel, time, bank)
            model.train()
            timedim = X_batch.size(2)
            # random_stride = np.random.random_integers(splice_frames//2, splice_frames)
            loss_sum = 0
            for i in range(0, timedim - splice_frames + 1, stride_frames):
                X = X_batch.narrow(2, i, splice_frames)
                y = y_batch
                if not config["no_cuda"]:
                    X = X.cuda()
                    y = y.cuda()
                optimizer.zero_grad()
                scores = model(X)
                loss = criterion(scores, y)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
                step_no += 1
                # learning rate change
            if epoch_idx > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"], momentum=config["momentum"],
                                            weight_decay=config["weight_decay"])
            if step_no % print_step == print_step -1:
                print_eval("train step #{}".format(step_no), scores, y, loss_sum/print_step, verbose=True)

        # evaluation on validation set
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            with torch.no_grad():
                model.eval()
                accs = []
                for X_batch, y_batch in dev_loader:
                    timedim = X_batch.size(2)
                    model_outputs = torch.tensor(0)
                    for i in range(0, timedim - splice_frames + 1, stride_frames):
                        X = X_batch.narrow(2, i, splice_frames)
                        y = y_batch
                        if not config["no_cuda"]:
                            X = X.cuda()
                            y = y.cuda()
                        optimizer.zero_grad()
                        if model_outputs.dim() == 0:
                            model_outputs = model.embed(X)
                        else:
                            model_outputs += model.embed(X)
                    agg_embed = model_outputs / (timedim // splice_frames)
                    scores = model.output(agg_embed)
                    loss = criterion(scores, y)
                    accs.append(print_eval("dev", scores, y, loss.item()))
                avg_acc = np.mean(accs)
                print("epoch #{}, dev accuracy: {}".format(epoch_idx,avg_acc))
                if avg_acc > max_acc:
                    print("saving best model...")
                    max_acc = avg_acc
                    model.save(config["output_file"])
    # test
    evaluate(config, model, test_loader)

def si_tdnn_train(config, loaders, model):
    train_loader, dev_loader, test_loader = loaders
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    # optimizer
    learnable_params = [param for param in model.parameters()
            if param.requires_grad == True]
    optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][0], nesterov=config["use_nesterov"],
                                weight_decay=config["weight_decay"], momentum=config["momentum"])
    criterion = nn.CrossEntropyLoss()

    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    max_acc = 0
    step_no = 0
    splice_frames = config["splice_frames"]
    stride_frames = config["stride_frames"]
    print_step = config["print_step"]

    # training iteration
    for epoch_idx in range(config["s_epoch"], config["n_epochs"]):
        model.train()
        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # X_batch = (batch, channel, time, bank)
            for i in range(0, X_batch.size(2)-(splice_frames+12), stride_frames):
                X = X_batch.narrow(2, i, splice_frames+12)
                y = y_batch
                if not config["no_cuda"]:
                    X = X.cuda()
                    y = y.cuda()
                optimizer.zero_grad()
                scores = model(X)
                # n_feat_per_seq = scores.size(1)
                # scores = scores.view(-1, scores.size(-1))
                # y = y.unsqueeze(1).expand(y.size(0), n_feat_per_seq)
                # y = y.contiguous().view(-1)
                loss = criterion(scores, y)
                loss.backward()
                optimizer.step()
            step_no += 1
            if step_no % print_step == print_step -1:
                print_eval("train step #{}".format(step_no), scores, y, loss.item(), verbose=True)

        # learning rate change
        if epoch_idx > schedule_steps[sched_idx]:
            sched_idx += 1
            print("changing learning rate to {}".format(config["lr"][sched_idx]))
            optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][sched_idx],
                                        nesterov=config["use_nesterov"],
                                        momentum=config["momentum"],
                                        weight_decay=config["weight_decay"])

        # evaluation on validation set
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            with torch.no_grad():
                model.eval()
                accs = []
                for X_batch, y_batch in tqdm(dev_loader, total=len(dev_loader)):
                    for i in range(0, X_batch.size(2)-(splice_frames+12), 1):
                        X = X_batch.narrow(2, i, splice_frames+12)
                        y = y_batch
                        if not config["no_cuda"]:
                            X = X.cuda()
                            y = y.cuda()
                        optimizer.zero_grad()
                        scores = model(X)
                        # n_feat_per_seq = scores.size(1)
                        # scores = scores.view(-1, scores.size(-1))
                        # y = y.unsqueeze(1).expand(y.size(0), n_feat_per_seq)
                        # y = y.contiguous().view(-1)
                        loss = criterion(scores, y)
                        accs.append(print_eval("dev", scores, y, loss.item()))

                avg_acc = np.mean(accs)
                print("epoch #{}, dev accuracy: {}".format(epoch_idx,avg_acc))
                if avg_acc > max_acc:
                    print("saving best model...")
                    max_acc = avg_acc
                    dir_name = os.path.dirname(config["output_file"])
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                    model.save(config["output_file"])


def si_full_train(config, loaders, model):
    train_loader, dev_loader, test_loader = loaders
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    # optimizer
    learnable_params = [param for param in model.parameters()
            if param.requires_grad == True]
    optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][0], nesterov=config["use_nesterov"],
                                weight_decay=config["weight_decay"], momentum=config["momentum"])
    criterion = nn.CrossEntropyLoss()

    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    max_acc = 0
    step_no = 0
    print_step = config["print_step"]

    # training iteration
    for epoch_idx in range(config["s_epoch"], config["n_epochs"]):
        model.train()
        for batch_idx, (X_batch, y_batch) in tqdm(enumerate(train_loader), total=len(train_loader)):
            # X_batch = (batch, channel, time, bank)
            X = X_batch
            y = y_batch
            if not config["no_cuda"]:
                X = X.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            scores = model(X)
            loss = criterion(scores, y)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no % print_step == print_step -1:
                print_eval("train step #{}".format(step_no), scores, y, loss.item(), verbose=True)

        # learning rate change
        if epoch_idx > schedule_steps[sched_idx]:
            sched_idx += 1
            print("changing learning rate to {}".format(config["lr"][sched_idx]))
            optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][sched_idx],
                                        nesterov=config["use_nesterov"],
                                        momentum=config["momentum"],
                                        weight_decay=config["weight_decay"])

        # evaluation on validation set
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            with torch.no_grad():
                model.eval()
                accs = []
                for X_batch, y_batch in tqdm(dev_loader, total=len(dev_loader)):
                    X = X_batch
                    y = y_batch
                    if not config["no_cuda"]:
                        X = X.cuda()
                        y = y.cuda()
                    optimizer.zero_grad()
                    scores = model(X)
                    loss = criterion(scores, y)
                    accs.append(print_eval("dev", scores, y, loss.item()))

                avg_acc = np.mean(accs)
                print("epoch #{}, dev accuracy: {}".format(epoch_idx,avg_acc))
                if avg_acc > max_acc:
                    print("saving best model...")
                    max_acc = avg_acc
                    dir_name = os.path.dirname(config["output_file"])
                    if not os.path.isdir(dir_name):
                        os.makedirs(dir_name)
                    model.save(config["output_file"])
