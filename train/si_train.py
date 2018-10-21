import numpy as np
import torch

from .train_utils import print_eval


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
            print("[{}/{}] train, loss: {:.4f}, acc: {:.5f} ".format(
                batch_idx, len(train_loader),
                loss_sum/(batch_idx+1), np.mean(accs)))

    avg_acc = np.mean(accs)

    return loss_sum, avg_acc

def length_schedule_train(config, train_loader, model, optimizer, criterion):
    model.train()
    loss_sum = 0
    accs = []

    print_steps = (np.arange(1,21) * 0.1
                    * len(train_loader)).astype(np.int64)

    splice_frames = config['splice_frames']
    if len(splice_frames) > 1:
        min_spFr, max_spFr = splice_frames
        # splice_frames_ =np.floor(max_spFr -  \
            # (max_spFr-min_spFr)*(config['epoch_idx']/config['n_epochs']))
        splice_frames_  =  max(max_spFr - 100 * np.floor(config['epoch_idx'] / 5),
                                min_spFr)
    else:
        splice_frames_ = splice_frames[-1]

    print("epoch {}, spFR: {}".format(config['epoch_idx'], splice_frames_))

    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.narrow(2, 0, int(splice_frames_))

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
            print("[{}/{}] train, loss: {:.4f}, acc: {:.5f} ".format(
                batch_idx, len(train_loader),
                loss_sum/(batch_idx+1), np.mean(accs)))

    avg_acc = np.mean(accs)

    return loss_sum, avg_acc


def batch_variable_length_train(config, train_loader, model, optimizer, criterion):
    model.train()
    loss_sum = 0
    accs = []

    print_steps = (np.arange(1,21) * 0.1
                    * len(train_loader)).astype(np.int64)



    for batch_idx, (X, y) in enumerate(train_loader):
        splice_frames = config['splice_frames']
        if len(splice_frames) > 1:
            splice_frames_ = np.random.randint(splice_frames[0], splice_frames[1])
            # min_spFr, max_spFr = splice_frames
            # # splice_frames_ =np.floor(max_spFr -  \
                # # (max_spFr-min_spFr)*(config['epoch_idx']/config['n_epochs']))
            # splice_frames_  =  max(max_spFr - 100 * np.floor(config['epoch_idx'] / 5),
                                    # min_spFr)
        else:
            splice_frames_ = splice_frames[-1]

        X = X.narrow(2, 0, int(splice_frames_))

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
            print("[{}/{}] train, loss: {:.4f}, acc: {:.5f} , spFr: {}".format(
                batch_idx, len(train_loader),
                loss_sum/(batch_idx+1), np.mean(accs), int(splice_frames_)))

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

def sub_utter_val(config, val_loader, model, criterion):
    n_corrects = 0
    total = 0
    with torch.no_grad():
        model.eval()
        loss_sum = 0
        splice_frames = config['splice_frames'][0]
        # stride_frames = config['stride_frames']

        for (X, y) in val_loader:
            if not config["no_cuda"]:
                X = X.cuda()
                y = y.cuda()

            split_points = range(0, X.size(2)-(splice_frames)+1, splice_frames)
            for point in split_points:
                x_in = X.narrow(2, point, splice_frames)
                scores = model(x_in)
                loss = criterion(scores, y)
                loss_sum += loss.item()
                n_corrects += torch.sum(torch.eq(torch.argmax(scores, dim=1), y)).item()
                total += y.size(0)
        avg_acc = n_corrects / total

        return loss_sum, avg_acc
