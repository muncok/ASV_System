import os
import numpy as np

from torch.autograd import Variable
import torch
import torch.nn as nn
import dnn.si_train as train

from dnn.parser import ConfigBuilder
from dnn.train import model as mod
from dnn.data import dataset as dset
import dnn.data.dataloader as dloader


model = "SimpleCNN"
dataset = "voxc"
global_config = dict(model=model, dataset=dataset,
                     no_cuda=False,  gpu_no=0,
                     n_epochs=100, batch_size=64,
                     lr=[0.01], schedule=[np.inf], dev_every=1, seed=0, use_nesterov=False,
                     cache_size=32768, momentum=0.9, weight_decay=0.00001,
                     num_workers=16, print_step=200,
                     )

builder = ConfigBuilder(
                dset.SpeechDataset.default_config(),
                global_config)
parser = builder.build_argparse()
si_config = builder.config_from_argparse(parser)
si_config['model_class'] = mod.SimpleCNN
train.set_seed(si_config)
### input
si_config['input_format'] = 'mfcc'
si_config['n_labels'] = 1881
si_config['input_length'] = int(16000*1)
si_config['time_dim'] = int(16000*0.1)//160+1

### model
si_model = si_config['model_class'](small=True)
# si_model.load_partial("models/voxc/si_voxc_0.1s_random.pt", fine_tune=True)
time_dim = si_config['time_dim']
test_in = Variable(torch.zeros(1,1,time_dim,40), volatile=True)
test_out = si_model(test_in)
si_model.feat_size = test_out.size(1)
si_model.output = nn.Linear(test_out.size(1), si_config["n_labels"])

### dataset
manifest_dir = "../interspeech2018/manifests/commands/"
si_config['train_manifest'] = os.path.join(manifest_dir,'si_train_manifest.csv')
si_config['val_manifest'] = os.path.join(manifest_dir,'si_val_manifest.csv')
si_config['test_manifest'] = os.path.join(manifest_dir,'si_test_manifest.csv')

## dataloaders
# words = ['stop', 'yes', 'seven', 'zero', 'up', 'no', 'two', 'go', 'four', 'one']
# train_loaders = []
# dev_loaders = []
# test_loaders = []
# for word in words:
#     for set_tag in ['train', 'val', 'test']:
#         si_config['{}_manifest'.format(set_tag)] = os.path.join(manifest_dir, 'si_{}_{}_manifest.csv'.format(word, set_tag))
#         # print(si_config['{}_manifest'.format(set_tag)])
#     train_loader, dev_loader, test_loader = dloader.get_loader(si_config)
#     train_loaders.append(train_loader); dev_loaders.append(dev_loader); test_loaders.append(test_loader)
# # train_loader = itertools.chain.from_iterable(train_loaders)
# # dev_loader = itertools.chain.from_iterable(dev_loaders)
# # test_loader = itertools.chain.from_iterable(test_loaders)
# loaders = [train_loaders, dev_loaders, test_loaders]

### train
# si_config['n_epochs'] = 50
# si_config['output_file'] = ("models/commands/batch_random.pt")
# train.train(si_config, model=si_model, loaders=None)

### pre-train
si_model.load("models/commands/batch_random.pt", fine_tune=True)
