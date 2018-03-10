import os
import numpy as np

import dnn.train.model as mod
from dnn.parser import ConfigBuilder
from dnn.si_train import set_seed, gatedcnn_train
from dnn.data.dataset import SpeechDataset

model = "res15"
dataset = "reddots"

global_config = dict(model=model, dataset=dataset,
                     no_cuda=False,  gpu_no=0,
                     n_epochs=30, batch_size=64,
                     lr=[1.0], schedule=[np.inf], dev_every=1, seed=0, use_nesterov=True,
                     cache_size=32768, momentum=0.90, weight_decay=0.00001,
                     num_workers=32, print_step=100,
                     )

builder = ConfigBuilder(
                mod.find_config(model),
                SpeechDataset.default_config(),
                global_config)
parser = builder.build_argparse()
si_config = builder.config_from_argparse(parser)
si_config['model_class'] = mod.find_model(model)
set_seed(si_config)

si_config['n_labels'] = 62
manifest_dir = "manifests/reddots/"
for tag in ['train', 'val', 'test']:
    si_config['{}_manifest'.format(tag)]=os.path.join(manifest_dir,'si_{}_{}_manifest.csv'.format("reddots", tag))

si_config['input_length'] = int(16000*3)
si_config['splice_length'] = int(16000*0.2)
si_config['input_format'] = 'mfcc'

# si_model = si_config['model_class'](si_config)

seq_len         = 20
embd_size       = si_config['n_mels']
n_layers        = 10
kernel          = (5, embd_size)
out_chs         = 64
res_block_count = 5
ans_size = si_config['n_labels']
si_model = mod.GatedCNN(seq_len, embd_size, n_layers, kernel, out_chs, res_block_count, ans_size)
si_config['input_file'] = ""
si_config['output_file'] = "models/reddots/si_reddots_gatedcnn.pt"
print(si_model)
gatedcnn_train(si_config, model=si_model)
