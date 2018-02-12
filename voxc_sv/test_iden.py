import argparse
import os
from data.data_loader import AudioDataLoader, SpectrogramDataset
import torch
from torch.autograd import Variable
from model import voxNet


parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--test_manifest', metavar='DIR',
                    help='path to test manifest csv', default='data/voxc_test_iden_manifest.csv')
parser.add_argument('--nb_class', default=1251, type=int, help='number of classes')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=16, type=int, help='Number of workers used in data-loading')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window_length', default=300, type=int, help='window_length')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')


def main():
    args = parser.parse_args()
    audio_conf = dict(sample_rate=args.sample_rate, window_size=args.window_size, window_stride=args.window_stride,
                      window=args.window)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.test_manifest,
                                            labels=None, normalize=True, augment=False)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model = voxNet(args.nb_class)
    checkpoint = torch.load(args.model_path)
    #model.load_state_dict(checkpoint['state_dict'])
    if args.model_path:
        if os.path.isfile(args.model_path):
            print("=> Loading checkpoint model %s" % args.model_path)
            checkpoint = torch.load(args.model_path)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> Loaded checkpoint model %s" % args.model_path)
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))
    model.cuda()
    nb_corrects = 0
    for j, (inputs, labels, input_percentage) in enumerate(test_loader):
        inputs,labels= Variable(inputs.cuda(), requires_grad=False),\
                             Variable(labels.cuda(), requires_grad=False)
        output = model(inputs)
        predicted = torch.max(output,1)[1] == labels
        nb_corrects += torch.sum(predicted).data[0]
    print("nb_corrects/total_input {}/{}, Acc:{}".format(nb_corrects, test_dataset.size,
                                                            nb_corrects / test_dataset.size))

if __name__ == '__main__':
    main()
