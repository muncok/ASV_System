import os, errno
import time
import argparse
import shutil

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR, StepLR, ExponentialLR

from model import voxNet, mVoxNet,weights_init, SpeechResModel
from data_loader import AudioDataLoader, SpectrogramDataset, MfeDataset, AudioDataLoaderMfe

parser = argparse.ArgumentParser(description='Speaker Identification Training')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/sv_reddot_okgoogle_train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/sv_reddot_okgoogle_val_manifest.csv')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to test manifest csv', default='data/sv_reddot_okgoogle_test_manifest.csv')
parser.add_argument('--dataset', default='voxceleb', help='voxceleb, ASVspoof, speech_command available')
parser.add_argument('--nb_class', default=190, type=int, help='number of classes')
parser.add_argument('--print_steps', default=20, type=int, help='loss print step (batches)')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=16, type=int, help='Number of workers used in data-loading')
parser.add_argument('--window_size', default=.025, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window_length', default=300, type=int, help='window_length')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--epochs', default=30, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--gpus', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=0.8532, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--log_dir', default='visualize/logs', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--save_folder', default='models/iden', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')

parser.add_argument('--noise_dir', default=None)
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--id', default='SI training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')


def to_np(x):
    return x.data.cpu().numpy()

def save_checkpoint(state, is_best, dir, filename='checkpoint.pth.tar'):
    save_path = os.path.join(dir,filename)
    torch.save(state, save_path)
    if is_best:
        best_state_path = os.path.join(dir, 'model_best.pth.tar')
        shutil.copyfile(save_path, best_state_path)

def main():
    # arguments & audio_conf
    args = parser.parse_args()
    audio_conf = dict(sample_rate=args.sample_rate, window_size=args.window_size, window_stride=args.window_stride,
                      window=args.window, noise_dir=args.noise_dir, noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    # dataset & data_loader
    dataset = args.dataset
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest,
                                                   labels=None, dataset=dataset, normalize=True, augment=args.augment)
    train_loader = AudioDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                    shuffle=True)

    val_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest,
                                                 labels=None, dataset=dataset, normalize=True, augment=args.augment)
    val_loader = AudioDataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest,
                                                  labels=None, dataset=dataset, normalize=True, augment=False)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

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
    model_config = {
            'n_labels' : args.nb_class ,
            'n_feature_maps' :  128, # {19, 45}
            'n_layers' : 6, #  {6, 13, 24}
            'use_dilation' : False
            }
    # model = SpeechResModel(model_config)
    model = mVoxNet(args.nb_class)
    # model.apply(weights_init)  # Xavier initialization
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.4)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5.0e-4, momentum=args.momentum)
    scheduler = ExponentialLR(optimizer, args.learning_anneal)

    if args.continue_from:
        if os.path.isfile(args.continue_from):
            print("=> Loading checkpoint model %s" % args.continue_from)
            checkpoint = torch.load(args.continue_from)
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint['batch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'best_val' in checkpoint:
                best_val =  checkpoint['best_val']
            print("=> Loaded checkpoint model %s" % args.continue_from)
        else:
            print("=> no checkpoint found at '{}'".format(args.continue_from))

    if 'best_val' not in locals():
        best_val = 0
    if 'start_epoch' not in locals():
        start_epoch = 0

    if args.gpus>1:
        model = nn.DataParallel(model)

    if args.cuda:
        model = model.cuda()

    seq_length = args.window_length
    print("Start Training")

    # train
    for epoch in range(start_epoch, args.epochs):
        model.train()
        scheduler.step()
        running_loss = 0.0
        train_loss_epoch = 0.0
        stime = time.time()
        # train a batch
        nb_corrects = 0
        for i, (inputs, labels) in enumerate(train_loader):
            if args.cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if args.cuda:
                torch.cuda.synchronize()

            nb_corrects += torch.sum(torch.max(outputs, 1)[1] == labels).data[0]
            running_loss += loss.data[0]
            train_loss_epoch += loss.data[0]
            if i % args.print_steps == (args.print_steps-1):
                etime = time.time()
                print('[%d, %5d] loss: %.3f, time: %f' %
                      (epoch + 1, i + 1, running_loss/args.print_steps, etime - stime))
                running_loss = 0.0
                stime = time.time()
            del outputs

        train_acc = nb_corrects / train_dataset.size
        print("[train] Acc: %.3f, corrects: %d" % (train_acc, nb_corrects))
        # end of epoch
        # validation
        nb_corrects = 0
        model.eval()
        for j, (inputs, labels) in enumerate(val_loader):
            if args.cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            if args.cuda:
                torch.cuda.synchronize()
            nb_corrects += torch.sum(torch.max(outputs, 1)[1] == labels).data[0]
        val_acc = nb_corrects / val_dataset.size
        print("[val] Acc: %.3f, corrects: %d" % (val_acc, nb_corrects))

        # save model
        if val_acc > best_val:
            best_val = val_acc
            is_best = True
        else:
            is_best = False

        if hasattr(model, 'module'):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        save_checkpoint({'epoch': epoch+1,
                         'batch': i+1,
                         'best_val': best_val,
                         'state_dict': state_dict,
                         'optimizer': optimizer.state_dict(),
                         }, is_best, args.save_folder, 'checkpoint_%d.ptr.tar'%(epoch+1))

        # tensorboard
        avg_train_loss = train_loss_epoch / train_dataset.size
        if args.tensorboard:
            optim_state = optimizer.state_dict()
            values = {
                'Avg Train loss': avg_train_loss,
                'Val_Acc':val_acc,
                'learning_rate':optim_state['param_groups'][0]['lr']
            }
            tensorboard_writer.add_scalars(args.id, values, epoch + 1)
            if args.log_params:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                    tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)


    print("Finished Training")

    model.eval()
    nb_corrects = 0
    for j, (inputs, labels) in enumerate(test_loader):
        if args.cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        nb_corrects += torch.sum(torch.max(outputs, 1)[1] == labels).data[0]
        del outputs
    acc = nb_corrects / test_dataset.size
    print("[test] Acc: %.3f, corrects: %d" % (acc, nb_corrects))

if __name__ == '__main__':
    main()
