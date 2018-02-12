import os, errno
import time
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, ExponentialLR
import torch.nn.functional as F
from torch.autograd import Variable

from model import siameseNet, weights_init
from data_loader import AudioDataLoaderPair, SpectrogramDatasetPair
from contrastive import ContrastiveLoss
from utils import  roc_auc_eer

parser = argparse.ArgumentParser(description='Speaker Verification training')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/sv_command_pair_manifest.csv')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to test manifest csv', default='data/sv_command_pair_manifest.csv')
parser.add_argument('--dataset', default='voxceleb', help='voxceleb, ASVspoof available')
parser.add_argument('--print_steps', default=20, type=int, help='loss print step (batches)')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--embedding', default=256, type=int, help='number of classes')
parser.add_argument('--num_workers', default=16, type=int, help='Number of workers used in data-loading')
parser.add_argument('--window_size', default=.025, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window_length', default=300, type=int, help='window_length')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--gpus', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=0.8532, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--log_dir', default='visualize/logs', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter testues and gradients')
parser.add_argument('--save_folder', default='models/veri', help='Location to save epoch models')
parser.add_argument('--model_path', default='/sv_best_final.pth.tar',
                    help='Location to save best testidation model')
parser.add_argument('--pretrain_path', default='models/model_best_300length.pth.tar',
                    help='pretrained model path')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')

parser.add_argument('--noise_dir', default=None)
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--id', default='SV training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')


def to_np(x):
    return x.data.cpu().numpy()

def save_checkpoint(state, is_best, dir, filename='checkpoint.pth.tar'):
    save_path = os.path.join(dir,filename)
    torch.save(state, save_path)
    if is_best:
        best_state_path = os.path.join(dir, 'model_best.pth.tar')
        shutil.copyfile(save_path, best_state_path)

def load_pretrained_model(model, pretrained_model_path):
    pretrained_model = torch.load(pretrained_model_path)
    pretrained_state_dict = pretrained_model['state_dict']
    state_dict = model.state_dict()
    transfered_layer = [model.conv1, model.conv2, model.conv3, model.conv4,
            model.conv5, model.fc6, model.fc7]

    # transfering weight including fc8
    for param in state_dict:
        if param in pretrained_state_dict and \
                pretrained_state_dict[param].size() == state_dict[param].size():  #TODO: fix it, it has problem
            state_dict[param] = pretrained_state_dict[param]
    model.load_state_dict(state_dict)

    # freeze layers except last fc layer
    for layer in transfered_layer:
        for param in layer.parameters():
            param.requires_grad = False

    print("load pretrained Model: Done")

    #names = [name for name in state_dict.keys() if 'running' not in name]
    #for name, param in zip(names, model.parameters()):
    #    print("{}: {}".format(name, param.requires_grad))
    return model



def main():
    # arguments & audio_conf
    args = parser.parse_args()
    audio_conf = dict(sample_rate=args.sample_rate, window_size=args.window_size, window_stride=args.window_stride,
                      window=args.window, noise_dir=args.noise_dir, noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))
    
    dataset = args.dataset
    # dataset & data_loader
    train_dataset = SpectrogramDatasetPair(audio_conf=audio_conf, manifest_filepath=args.train_manifest,
                                                   labels=None, dataset=dataset, normalize=True, augment=args.augment)
    train_loader = AudioDataLoaderPair(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                    shuffle=True)
    test_dataset = SpectrogramDatasetPair(audio_conf=audio_conf, manifest_filepath=args.test_manifest,
                                                  labels=None, dataset=dataset, normalize=True, augment=False)
    test_loader = AudioDataLoaderPair(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # tensorboard
    if args.tensorboard:
        try:
            os.makedirs(args.log_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Tensorboard log directory already exists.')
                for file in os.listdir(args.log_dir):
                    file_path = os.path.join(args.log_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        raise
            else:
                raise
        from tensorboardX import SummaryWriter
        tensorboard_writer = SummaryWriter(args.log_dir)

    # model save folder
    try:
        os.makedirs(args.save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Model Save directory already exists.')
        else:
            raise

    # model
    model = siameseNet(args.embedding)
    model.apply(weights_init)
    model = load_pretrained_model(model, args.pretrain_path)
    criterion = ContrastiveLoss()
    learnable_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = optim.SGD(learnable_params, lr=args.lr, momentum=0.9, weight_decay=5.0e-4)
    #optimizer = optim.Adam(learnable_params, lr=0.01, weight_decay=5.0e-4)
    #scheduler = LambdaLR(optimizer, lambda epoch: 0.95**epoch)
    scheduler = ExponentialLR(optimizer, args.learning_anneal)

    if args.continue_from:
        if os.path.isfile(args.continue_from):
            print("=> Loading checkpoint model %s" % args.continue_from)
            checkpoint = torch.load(args.continue_from)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'best_eer' in checkpoint:
                best_eer =  checkpoint['best_eer']
            print("=> Loaded checkpoint model %s" % args.continue_from)
        else:
            print("=> no checkpoint found at '{}'".format(args.continue_from))

    if 'best_eer' not in locals():
        best_eer = 1
    if 'start_epoch' not in locals():
        start_epoch = 0

    if args.gpus>1:
        model = nn.DataParallel(model)

    if args.cuda:
        model = model.cuda()

    seq_length = args.window_length
    print_step = args.print_steps
    print("Start Training")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        scheduler.step()
        running_loss = 0.0
        train_dists = np.empty(0)
        label_list = np.empty(0)
        stime = time.time()
        for i, ((input0, input1), labels) in enumerate(train_loader):
            input0,input1,labels= Variable(input0.cuda(), requires_grad=False), \
                                   Variable(input1.cuda(), requires_grad=False), \
                                   Variable(labels.cuda(), requires_grad=False)
            optimizer.zero_grad()
            output0, output1 = model(input0, input1)
            loss = criterion(output0, output1, labels)
            loss.backward()
            optimizer.step()
            if args.cuda:
                torch.cuda.synchronize()

            running_loss += loss.data[0]
            train_dist = -F.pairwise_distance(output0, output1)
            train_dists = np.append(train_dists, to_np(train_dist))
            label_list = np.append(label_list, to_np(labels))
            if i % args.print_steps == (args.print_steps-1):
                etime = time.time()
                auc, eer = roc_auc_eer(train_dists, label_list)
                print('[%d, %5d] loss: %.3f, auc: %.2f, eer: %.2f, time: %f' %
                      (epoch + 1, i + 1, running_loss/print_step, auc, eer, etime - stime))
                running_loss = 0.0
                train_dists = np.empty(0)
                label_list = np.empty(0)
                stime = time.time()
            del output0, output1, train_dist

        # each epoch
        # validation
        model.eval()
        test_loss = 0
        test_dists = np.empty(0)
        label_list = np.empty(0)
        for j, ((input0, input1), labels) in enumerate(test_loader):
            if args.cuda:
                input0,input1,labels= Variable(input0.cuda()), Variable(input1.cuda()), \
                                       Variable(labels.cuda())
            else:
                input0,input1,labels= Variable(input0), Variable(input1), \
                                       Variable(labels)
            output0, output1 = model(input0, input1)
            loss = criterion(output0, output1, labels)
            test_loss += loss.data[0]
            test_dist = F.cosine_similarity(output0, output1)
            test_dists = np.append(test_dists, to_np(test_dist))
            label_list = np.append(label_list, to_np(labels))
            del output0, output1, test_dist
        test_auc, test_eer = roc_auc_eer(test_dists, label_list)
        print('[test] loss: %.3f, auc: %.2f, eer: %.2f'%(test_loss/test_dataset.size, test_auc, test_eer))
        is_best = test_eer < best_eer
        if is_best:
            best_eer = test_eer
        save_checkpoint({'epoch': epoch+1,
                         'state_dict': model.state_dict(),
                         'best_eer': best_eer,
                         'optimizer': optimizer.state_dict(),
                         }, is_best, args.save_folder, 'checkpoint_%d.ptr.tar'%(epoch+1))
    print("Finished Training")

    model.eval()
    test_loss = 0
    test_dists = np.empty(0)
    label_list = np.empty(0)
    for j, ((input0, input1), labels) in enumerate(test_loader):
        input0,input1,labels= Variable(input0.cuda(), requires_grad=False), \
                               Variable(input1.cuda(), requires_grad=False), \
                               Variable(labels.cuda(), requires_grad=False)
        output0, output1 = model(input0, input1)
        test_dist = -F.pairwise_distance(output0, output1)
        test_dists = np.append(test_dists, to_np(test_dist))
        label_list = np.append(label_list, to_np(labels))
        loss = criterion(output0, output1, labels)
        test_loss += loss.data[0]
        del output0, output1, test_dist, labels
    test_auc, test_eer = roc_auc_eer(test_dists, label_list)
    print('[test] loss: %.3f, auc: %.2f, eer: %.2f'%(test_loss/test_dataset.size, test_auc, test_eer))

if __name__ == '__main__':
    main()
