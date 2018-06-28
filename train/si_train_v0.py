import numpy as np

import torch
import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from .train_utils import print_eval

def evaluate(config, model, test_loader, tqdm_v=tqdm):
    with torch.no_grad():
        model.eval()
        accs = []
        for (X, y) in tqdm_v(test_loader, total=len(test_loader)):
            if not config["no_cuda"]:
                X = X.cuda()
                y = y.cuda()
            scores = model(X)
            # no loss value
            accs.append(print_eval("test", scores, y, 0))
        avg_acc = np.mean(accs)
    print("final test accuracy: {}".format(avg_acc))


def si_train(config, loaders, model, criterion = nn.CrossEntropyLoss(), tqdm_v=tqdm):
    train_loader, dev_loader, test_loader = loaders
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    # optimizer
    learnable_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][0], nesterov=config["use_nesterov"],
                                weight_decay=config["weight_decay"], momentum=config["momentum"])
    criterion_ = criterion

    schedule_steps = config["schedule"]; schedule_steps.append(np.inf)
    sched_idx = 0; max_acc = 0; step_no = 0
    print_step = config["print_step"]

    # training iteration
    for epoch_idx in range(config["s_epoch"], config["n_epochs"]):
        model.train()
        loss_sum = 0
        for batch_idx, (X, y) in tqdm_v(enumerate(train_loader),
            total=len(train_loader)):
            # X_batch = (batch, channel, time, bank)
            if not config["no_cuda"]:
                X = X.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            scores = model(X)
            loss = criterion_(scores, y)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            step_no += 1
            # learning rate change
            if epoch_idx > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}"
                        .format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(learnable_params,
                        lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"],
                    momentum=config["momentum"],
                    weight_decay=config["weight_decay"])
                model.save(config["output_file"].rstrip('.pt')+".{}.pt".format(
                            config["lr"][sched_idx]))
            if step_no % print_step == print_step -1:
                # schedule over iteration
                print_eval("train step #{}".format(step_no), scores, y,
                        loss_sum/print_step, verbose=True)

        # evaluation on validation set
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            with torch.no_grad():
                model.eval()
                accs = []
                for (X, y) in tqdm_v(dev_loader,
                        total=len(dev_loader)):
                    if not config["no_cuda"]:
                        X = X.cuda()
                        y = y.cuda()
                    scores = model(X)
                    loss = criterion_(scores, y)
                    accs.append(print_eval("dev", scores, y,
                        loss.item()))
                avg_acc = np.mean(accs)
                print("epoch #{}, dev accuracy: {}".format(epoch_idx,
                    avg_acc))
                if avg_acc > max_acc:
                    print("saving best model...")
                    max_acc = avg_acc
                    model.save(config["output_file"]+".pt")
    # test
    evaluate(config, model, test_loader)

