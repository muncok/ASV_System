import argparse
from data_loader import AudioDataLoaderPair, SpectrogramDatasetPair
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from model import siameseNet
from utils import  roc_auc_eer, to_np


parser = argparse.ArgumentParser(description='Speaker Verification Test')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to test manifest csv', default='data/sv_command_pair_manifest.csv')
parser.add_argument('--nb_class', default=2, type=int, help='number of classes')
parser.add_argument('--print_steps', default=20, type=int, help='loss print step (batches)')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window_length', default=300, type=int, help='window_length')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--model_path', default='models/iden/model.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--noise_dir', default=None)
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--threshold', default=0.1, help='threshold for classification', type=float)


def main():
    args = parser.parse_args()
    audio_conf = dict(sample_rate=args.sample_rate, window_size=args.window_size, window_stride=args.window_stride,
                      window=args.window, noise_dir=args.noise_dir, noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))
    test_dataset = SpectrogramDatasetPair(audio_conf=audio_conf, manifest_filepath=args.test_manifest,
                                            labels=None, normalize=True, augment=False)
    test_loader = AudioDataLoaderPair(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model = siameseNet(args.nb_class)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    if args.cuda:
        model.cuda()
    model.eval()
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
        del output0, output1, test_dist, labels
    test_auc, test_eer = roc_auc_eer(test_dists, label_list)
    print('[test] auc: %.2f, eer: %.2f'%(test_auc, test_eer))

if __name__ == '__main__':
    main()
