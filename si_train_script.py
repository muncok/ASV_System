import os
import numpy as np

from dnn import si_train
from dnn.train import model as mod
from dnn.data import dataset as dset
from dnn.data import dataloader as dloader
from dnn.parser import ConfigBuilder

model = "res15"
dataset = "voxc"

global_config = dict(model=model, dataset=dataset,
                     no_cuda=False,  gpu_no=0,
                     n_epochs=100, batch_size=64,
                     lr=[0.01], schedule=[np.inf], dev_every=1, seed=0, use_nesterov=False,
                     cache_size=32768, momentum=0.9, weight_decay=0.00001,
                     num_workers=16, print_step=np.inf,
                     )

builder = ConfigBuilder(
                mod.find_config(model),
                dset.SpeechDataset.default_config(),
                global_config
            )

parser = builder.build_argparse()
si_config = builder.config_from_argparse(parser)
si_config['model_class'] = mod.find_model(model)
si_train.set_seed(si_config)

si_config['n_labels'] = 1260
manifest_dir = "../interspeech2018/manifests/voxc/"
si_config['train_manifest'] = os.path.join(manifest_dir,'si_voxc_train_manifest.csv')
si_config['val_manifest'] = os.path.join(manifest_dir,'si_voxc_val_manifest.csv')
si_config['test_manifest'] = os.path.join(manifest_dir,'si_voxc_test_manifest.csv')

from torch.autograd import Variable
import torch
import torch.nn as nn

si_model = si_config['model_class'](si_config)
si_config['input_length'] = int(16000*3)
si_config['splice_dim'] = int(16000*0.1)//160+1
# time_dim = si_config['splice_dim']
# test_in = Variable(torch.zeros(1,1,time_dim,40), volatile=True)
# test_out = si_model(test_in)
# si_model.feat_size = test_out.size(1)
# si_model.output = nn.Linear(test_out.size(1), si_config["n_labels"])

si_config['n_epochs'] = 50
si_config['input_format'] = 'mfcc'
si_config['output_file'] = ("models/voxc/si_train/full_train/si_voxc_res15_0.1s_full.pt")
loaders = dloader.get_loader(si_config, datasets=None)
si_train.train(si_config, model=si_model, loaders=loaders)

