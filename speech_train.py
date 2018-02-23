# coding=utf-8
import numpy as np
import scipy.stats as st
from tqdm import tqdm
from time import sleep

import torch
from torch.autograd import Variable
import honk_sv.model as mod

from protonet.prototypical_batch_sampler import PrototypicalBatchSampler
from protonet.verification_batch_sampler import VerificationBatchSampler
from protonet.prototypical_loss import prototypical_loss as loss
from protonet.verification_loss import verification_score as veri_score
from protonet.speech_dataset import SpeechDataset
from protonet.parser import get_parser


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_dataset(opt):
    '''
    Initialize the datasets, samplers and dataloaders
    '''
    train_dataset, _, val_dataset = SpeechDataset.read_train_manifest(opt)

    tr_sampler = PrototypicalBatchSampler(labels=train_dataset.audio_labels,
                                          classes_per_it=opt.classes_per_it_tr,
                                          num_samples=opt.num_support_tr + opt.num_query_tr,
                                          iterations=opt.iterations,
                                          randomize=False)

    # trainval_sampler = PrototypicalBatchSampler(labels=trainval_dataset.audio_labels,
                                                # classes_per_it=opt.classes_per_it_tr,
                                                # num_samples=opt.num_support_tr + opt.num_query_tr,
                                                # iterations=opt.iterations,
                                                # randomize=False)

    # val_sampler = PrototypicalBatchSampler(labels=val_dataset.audio_labels,
                                           # classes_per_it=opt.classes_per_it_val,
                                           # num_samples=opt.num_support_val + opt.num_query_val,
                                           # iterations=opt.iterations,
                                           # randomize=False)

    val_sampler = VerificationBatchSampler(labels=val_dataset.audio_labels,
                                           classes_per_it=opt.classes_per_it_val,
                                           num_support=opt.num_support_val,
                                           num_query=opt.num_query_val,
                                           iterations=opt.iterations)

    tr_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_sampler=tr_sampler,
                                                num_workers=16)

    # trainval_dataloader = torch.utils.data.DataLoader(trainval_dataset,
                                                      # batch_sampler=trainval_sampler)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_sampler=val_sampler)


    return tr_dataloader, val_dataloader

def init_speechnet(opt):

    import model as mod
    model_name = "cnn-small"
    config = mod.find_config(model_name)
    config["n_labels"] = opt.classes_per_it_tr
    model_class = mod.find_model(model_name)
    model = model_class(config)
    model = model.cuda() if opt.cuda else model
    return model

def init_protonet(opt, model_path=None, fine_tune=False):
    '''
    Initialize the pre-trained resnet
    '''
    model = mod.SimpleCNN()

    if model is not None:
        to_state = model.state_dict()
        from_state = torch.load(model_path)
        valid_state = {k:v for k,v in from_state.items() if k in to_state.keys()}
        to_state.update(valid_state)
        model.load_state_dict(to_state)
        print("{} is loaded".format(model_path))

    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.convb_4.parameters():
            param.requires_grad = True

    model = model.cuda() if opt.cuda else model
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(model.convb_4.parameters(), lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optim,
                                           step_size=opt.lr_scheduler_step,
                                           gamma=opt.lr_scheduler_gamma)


def train(opt, tr_dataloader, val_dataloader, model, optim, lr_scheduler):
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
        for batch in tqdm(tr_iter):
            x, y = batch
            x, y = Variable(x), Variable(y)
            if opt.cuda:
                x, y = x.cuda(), y.cuda()
            model_output = model(x)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
            l.backward()
            optim.step()
            train_loss.append(l.data[0])
            train_acc.append(acc.data[0])
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Train Loss: {}, Train Acc: {}'.format(avg_loss, avg_acc))
        val_iter = iter(val_dataloader)
        for batch in tqdm(val_iter):
            x, y = batch
            x, y = Variable(x), Variable(y)
            if opt.cuda:
                x, y = x.cuda(), y.cuda()
            model_output = model(x)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
            val_loss.append(l.data[0])
            val_acc.append(acc.data[0])
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc > best_acc else ''
        print('Val Loss: {}, Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
        if avg_acc > best_acc:
            torch.save(model.state_dict(), opt.output)
            best_acc = avg_acc

        lr_scheduler.step()

    return best_acc, train_loss, train_acc, val_loss, val_acc

def evaluate(opt, val_dataloader, model):
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
        val_loss.append(l.data[0])
        val_acc.append(acc.data[0])
    avg_loss = np.mean(val_loss[-opt.iterations:])
    avg_acc = np.mean(val_acc[-opt.iterations:])
    print('Val Loss: {}, Val Acc: {}'.format(avg_loss, avg_acc))

def sv_score(opt, val_dataloader, model):
    eer_records = []
    val_iter = iter(val_dataloader)
    for batch in tqdm(val_iter):
        x, y = batch
        x, y = Variable(x), Variable(y)
        if opt.cuda:
            x, y = x.cuda(), y.cuda()
        model_output = model(x)
        eer = veri_score(model_output, target=y, n_classes=opt.classes_per_it_val,
                               n_support=opt.num_support_val, n_query=opt.num_query_val)
        # print("eer:{:.2f}%".format(eer*100))
        eer_records.append(eer)
    sleep(0.05)
    mean_eer = np.mean(eer_records)
    lb, ub = st.t.interval(0.95, len(eer_records)-1, loc=mean_eer, scale=st.sem(eer_records))
    print("eer: {:.2f}% +- {:.2f}%".format(mean_eer*100, (ub-mean_eer)*100))


def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    options.train_manifest = "manifests/reddots/fewshot/si_reddots_train_manifest.csv"
    options.val_manifest = "manifests/reddots/fewshot/si_reddots_val_manifest.csv"
    options.test_manifest = "manifests/reddots/fewshot/si_reddots_test_manifest.csv"
    options.n_dct_filters = 40
    options.input_length = 16000
    options.n_mels = 40
    options.timeshift_ms = 100
    options.data_folder = "/home/muncok/DL/dataset/SV_sets"
    options.window_size= 0.025
    options.window_stride= 0.010
    options.cache_size = 50000

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    tr_dataloader, val_dataloader = init_dataset(options)

    if options.mode == "train":
        #### train ####
        model = init_protonet(options)
        # model = init_protonet(options, "models/reddots/fewshot/si_reddots_frames_fewshot.pt")
        optim = init_optim(options, model)
        lr_scheduler = init_lr_scheduler(options, optim)
        train(opt=options,
              tr_dataloader=tr_dataloader,
              val_dataloader=val_dataloader,
              model=model,
              optim=optim,
              lr_scheduler=lr_scheduler)
    elif options.mode == "eval":
        ### evaluate ####
        model = init_protonet(options, "models/many_way.pth")
        evaluate(options, val_dataloader, model)
        # model = init_protonet(options, "models/reddots/fewshot/si_reddots_frames_fewshot.pt")
        # evaluate(options, val_dataloader, model)
    elif options.mode == "sv_score":
        ### verification scoring ####
        model = init_protonet(options, "models/many_way.pth")
        sv_score(options, val_dataloader, model)
        model = init_protonet(options, "models/reddots/fewshot/si_reddots_frames_fewshot.pt")
        sv_score(options, val_dataloader, model)

if __name__ == '__main__':
    main()
