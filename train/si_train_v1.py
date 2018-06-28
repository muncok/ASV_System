import numpy as np
import pandas as pd
import torch.nn.functional as F


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from .train_utils import print_eval

from sv_system.data.dataset import find_dataset
from sv_system.sv_score.score_utils import embeds_utterance
from sv_system.data.dataloader import init_default_loader
from sklearn.metrics import roc_curve

def si_train(config, loaders, model, criterion = nn.CrossEntropyLoss(), tqdm_v=tqdm):
    train_loader, dev_loader, test_loader = loaders
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    # optimizer
    learnable_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][0], nesterov=config["use_nesterov"],
                                weight_decay=config["weight_decay"], momentum=config["momentum"])
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0.001, patience=10)
    criterion_ = criterion

    # max_acc = 0
    step_no = 0
    print_step = config["print_step"]

    # sv_score
    voxc_test_df = pd.read_pickle("dataset/dataframes/voxc/sv_voxc_dataframe.pkl")
    _, dset, n_labels = find_dataset(config, config['dataset'])
    voxc_test_dset = dset.read_df(config, voxc_test_df, "test")
    val_dataloader = init_default_loader(config, voxc_test_dset, shuffle=False)
    trial = pd.read_pickle("dataset/dataframes/voxc/voxc_trial.pkl")
    cord = [trial.enrolment_id.tolist(), trial.test_id.tolist()]
    label_vector = trial.label
    min_eer = 1

    # training iteration
    prev_lr = config["lr"][0]
    lr_change_cnt = 0
    for epoch_idx in range(config["s_epoch"], config["n_epochs"]):
        loss_sum = 0
        input_frames = np.random.randint(300, 800)
        train_loader.dataset.input_frames = input_frames
        model.train()
        for batch_idx, (X, y) in tqdm_v(enumerate(train_loader),
                total=len(train_loader)):
            # X_batch = (batch, channel, time, bank)
            # X = X_batch.narrow(2, 0, input_frames)
            if not config["no_cuda"]:
                X = X.cuda()
                y = y.cuda()
            optimizer.zero_grad()
            scores = model(X)
            loss = criterion_(scores, y)
            loss_sum += loss.item()
            loss.backward()
            # learning rate change
            optimizer.step()
            step_no += 1
            if step_no % print_step == print_step -1:
                # schedule over iteration
                print_eval("train step #{}".format(step_no), scores, y,
                        loss_sum/(batch_idx+1), verbose=True)
            input_frames = np.random.randint(300, 800)
            train_loader.dataset.input_frames = input_frames
        # change lr accoring to training loss
        scheduler.step(loss_sum)
        curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print("epoch #{}, train loss: {}, lr: {}".format(epoch_idx,
            loss_sum, curr_lr))
        if prev_lr == curr_lr:
            lr_change_cnt += 1
            print("saving model before the lr changed")
            prev_lr = curr_lr
            model.save(config["output_file"].rstrip('.pt')+".{}.pt".format(lr_change_cnt))

        # evaluation on validation set
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            # with torch.no_grad():
                # model.eval()
                # accs = []
                # loss_sum = 0
                # dev_loader.dataset.input_frames = 500
                # for (X, y) in tqdm_v(dev_loader,
                        # total=len(dev_loader)):
                    # # X = X_batch.narrow(2, 0, input_frames)
                    # if not config["no_cuda"]:
                        # X = X.cuda()
                        # y = y.cuda()
                    # scores = model(X)
                    # loss = criterion_(scores, y)
                    # loss_sum += loss.item()
                    # accs.append(print_eval("dev", scores, y,
                        # loss.item()))
                # avg_acc = np.mean(accs)
                # print("epoch #{}, dev accuracy: {}".format(epoch_idx,
                    # avg_acc))
                # if avg_acc > max_acc:
                    # print("saving best model...")
                    # max_acc = avg_acc
                    # model.save(config["output_file"])
                embeddings, _ = embeds_utterance(config, val_dataloader, model, None)
                sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
                score_vector = sim_matrix[cord].numpy()
                fpr, tpr, thres = roc_curve(
                        label_vector, score_vector, pos_label=1)
                eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
                print("epoch #{}, dev eer: {}".format(epoch_idx,
                    eer))
                if eer < min_eer:
                    print("saving best model...")
                    min_eer = eer
                    model.save(config["output_file"])
    # test
    with torch.no_grad():
        model.eval()
        accs = []
        loss_sum = 0
        for (X, y) in tqdm_v(test_loader,
                total=len(test_loader)):
            test_loader.dataset.input_frames = 800
            if not config["no_cuda"]:
                X = X.cuda()
                y = y.cuda()
            scores = model(X)
            loss = criterion_(scores, y)
            loss_sum += loss.item()
            accs.append(print_eval("test", scores, y,
                loss.item()))
        avg_acc = np.mean(accs)
        print("test accuracy: {}".format(avg_acc))
