
# coding=utf-8
import numpy as np
from tqdm import tqdm
from time import sleep

import torch
from torch.autograd import Variable

def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    learnable_parameters = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.Adam(learnable_parameters, lr=opt.learning_rate)

def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optim,
                                           step_size=opt.lr_scheduler_step,
                                           gamma=opt.lr_scheduler_gamma)

def train(opt, tr_dataloader, val_dataloader, model, optim, lr_scheduler, loss):
    '''
    Train the model with the prototypical learning algorithm
    '''
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)

        #### full scan
        cnt = 0
        for batch in tqdm(tr_iter):
            x, y = batch
            y = Variable(y)
            if opt.cuda:
                y = y.cuda()
            time_dim = x.size(2)
            splice_dim = opt.splice_dim
            split_points = range(0, time_dim-splice_dim+1, splice_dim)
            for point in split_points:
                x_in = Variable(x.narrow(2, point, splice_dim))
                if opt.cuda:
                    x_in = x_in.cuda()
                model_output = model(x_in)
                l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
                l.backward()
                optim.step()
                train_loss.append(l.data[0])
                train_acc.append(acc.data[0])
                cnt += 1
        avg_loss = np.mean(train_loss[-cnt:])
        avg_acc = np.mean(train_acc[-cnt:])
        print('Train Loss: {}, Train Acc: {}'.format(avg_loss, avg_acc))
        sleep(0.05)
        val_iter = iter(val_dataloader)
        for batch in tqdm(val_iter):
            x, y = batch
            y = Variable(y)
            if opt.cuda:
                y = y.cuda()
            time_dim = x.size(2)
            splice_dim = opt.splice_dim
            split_points = range(0, time_dim-splice_dim+1, splice_dim)
            for point in split_points:
                x_in = Variable(x.narrow(2, point, splice_dim))
                if opt.cuda:
                    x_in = x_in.cuda()
                model_output = model(x_in)
                l, acc = loss(model_output, target=y, n_support=opt.num_support_val)
                val_loss.append(l.data[0])
                val_acc.append(acc.data[0])
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc > best_acc else '(Best was {})'.format(best_acc)
        print('Val Loss: {}, Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
        if avg_acc > best_acc:
            torch.save(model.state_dict(), opt.output)
            best_acc = avg_acc

        lr_scheduler.step()

    return best_acc, train_loss, train_acc, val_loss, val_acc

def evaluate(opt, val_dataloader, model, loss):
    val_loss = []
    val_acc = []
    val_iter = iter(val_dataloader)
    for batch in tqdm(val_iter):
        x, y = batch
        x, y = Variable(x), Variable(y)
        if opt.cuda:
            x, y = x.cuda(), y.cuda()
        model_output = model(x)
        l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
        # l, acc = loss(model_output, target=y, n_classes=opt.classes_per_it_val,
        #                 n_support=opt.num_support_tr, n_query=opt.num_query_val)
        val_loss.append(l.data[0])
        val_acc.append(acc.data[0])
    avg_loss = np.mean(val_loss[-opt.iterations:])
    avg_acc = np.mean(val_acc[-opt.iterations:])
    print('Val Loss: {}, Val Acc: {}'.format(avg_loss, avg_acc))

