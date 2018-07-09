import numpy as np
import pandas as pd
import torch.nn.functional as F

import torch
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm
from .train_utils import (print_eval, save_checkpoint, get_dir_path,
        load_checkpoint)

from data.dataset import featDataset
from sv_score.score_utils import embeds_utterance
from data.dataloader import init_default_loader
from sklearn.metrics import roc_curve

from tensorboardX import SummaryWriter

"""
MultiStep LR scheduler
"""

def si_train(config, loaders, model, optimizer, criterion, tqdm_v=tqdm):
    log_dir = get_dir_path(config['output_file'])
    writer = SummaryWriter(log_dir)
    train_loader, dev_loader, test_loader = loaders

    if config['input_file']:
        load_checkpoint(config, model, optimizer)

    scheduler = MultiStepLR(optimizer, milestones=config['lr_schedule'], gamma=0.1,
            last_epoch=config['s_epoch']-1)

    # max_acc = 0
    step_no = 0
    print_step = config["print_step"]

    # sv_score
    voxc_test_df = pd.read_pickle("dataset/dataframes/voxc/sv_voxc_dataframe.pkl")
    voxc_test_dset = featDataset.read_df(config, voxc_test_df, "test")
    val_dataloader = init_default_loader(config, voxc_test_dset, shuffle=False)
    trial = pd.read_pickle("dataset/dataframes/voxc/voxc_trial.pkl")
    cord = [trial.enrolment_id.tolist(), trial.test_id.tolist()]
    label_vector = np.array(trial.label)
    min_eer = config['best_metric'] if 'best_metric' in config else 1.0

    # training iteration
    for epoch_idx in range(config["s_epoch"], config["n_epochs"]):
        # learning rate change
        scheduler.step()
        curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print("curr_lr: {}".format(curr_lr))
        writer.add_scalar("train/lr", curr_lr, epoch_idx)

        loss_sum = 0
        input_frames = np.random.randint(300, 800)
        train_loader.dataset.input_frames = input_frames

        if len(config['gpu_no']) > 1:
            model = torch.nn.DataParallel(model)

        model.train()
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch_idx)

        accs = []
        for batch_idx, (X, y) in tqdm_v(enumerate(train_loader), ncols=100,
                total=len(train_loader)):
            # X_batch = (batch, channel, time, bank)
            # X = X_batch.narrow(2, 0, input_frames)
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
            step_no += 1
            if step_no % print_step == print_step -1:
                # schedule over iteration
                accs.append(print_eval("train step #{}".format(step_no), scores, y,
                        loss_sum/(batch_idx+1), verbose=True))
            input_frames = np.random.randint(300, 800)
            train_loader.dataset.input_frames = input_frames

        avg_acc = np.mean(accs)
        # tensorboard
        writer.add_scalar('train/loss', loss_sum, epoch_idx)
        writer.add_scalar('train/acc', avg_acc, epoch_idx)
        # change lr accoring to training loss
        print("epoch #{}, train loss: {}, lr: {}".format(epoch_idx,
            loss_sum, curr_lr))

        # evaluation on validation set
        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            with torch.no_grad():
                model.eval()
                accs = []
                loss_sum = 0
                dev_loader.dataset.input_frames = 500
                for (X, y) in tqdm_v(dev_loader, ncols=100,
                        total=len(dev_loader)):
                    # X = X_batch.narrow(2, 0, input_frames)
                    if not config["no_cuda"]:
                        X = X.cuda()
                        y = y.cuda()
                    scores = model(X)
                    loss = criterion(scores, y)
                    loss_sum += loss.item()
                    accs.append(print_eval("dev", scores, y,
                        loss.item()))
                avg_acc = np.mean(accs)
                # tensorboard
                writer.add_scalar('dev/loss', loss_sum, epoch_idx)
                writer.add_scalar('dev/acc', avg_acc, epoch_idx)

                if isinstance(model, torch.nn.DataParallel):
                    model = model.module

                # compute eer
                embeddings, _ = embeds_utterance(config, val_dataloader, model, None)
                sim_matrix = F.cosine_similarity(
                        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
                score_vector = sim_matrix[cord].numpy()
                fpr, tpr, thres = roc_curve(
                        label_vector, score_vector, pos_label=1)
                eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
                writer.add_scalar('dev/sv_eer', eer, epoch_idx)
                writer.add_pr_curve('DET', label_vector, score_vector, epoch_idx)

                print("epoch #{}, dev accuracy: {}".format(epoch_idx, avg_acc))
                print("epoch #{}, dev eer: {}".format(epoch_idx, eer))

                if eer < min_eer:
                    min_eer = eer
                    is_best = True
                else:
                    is_best = False
                filename = get_dir_path(config["output_file"]) + \
                "/model.{:.4}.pt.tar".format(curr_lr)
                save_checkpoint({
                    'epoch': epoch_idx,
                    'arch': config["arch"],
                    'state_dict': model.state_dict(),
                    'best_metric': eer,
                    'optimizer' : optimizer.state_dict(),
                    }, epoch_idx, is_best, filename=filename)
    # test
    with torch.no_grad():
        model.eval()
        accs = []
        loss_sum = 0
        for (X, y) in tqdm_v(test_loader, ncols=100,
                total=len(test_loader)):
            test_loader.dataset.input_frames = 800
            if not config["no_cuda"]:
                X = X.cuda()
                y = y.cuda()
            scores = model(X)
            loss = criterion(scores, y)
            loss_sum += loss.item()
            accs.append(print_eval("test", scores, y,
                loss.item()))
        avg_acc = np.mean(accs)
        print("test accuracy: {}".format(avg_acc))
        writer.add_scalar('test/acc', avg_acc, 0)

    writer.close()