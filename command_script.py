import os
import numpy as np

import honk_sv.model as mod
import honk_sv.train as hk

model = "cnn-frames"
dataset = "command"

global_config = dict(model=model, dataset=dataset,
                     no_cuda=False,  gpu_no=0,
                     n_epochs=30, batch_size=64,
                     lr=[0.001], schedule=[np.inf], dev_every=1, seed=0, use_nesterov=False,
                     cache_size=32768, momentum=0.9, weight_decay=0.00001,
                     num_workers=32, print_step=1,
                     splice_length = 9
                     )

builder = hk.ConfigBuilder(
                mod.find_config(model),
                mod.SpeechDataset.default_config(dataset),
                global_config)
parser = builder.build_argparse()
si_config = builder.config_from_argparse(parser)
si_config['model_class'] = mod.find_model(model)
hk.set_seed(si_config)

si_config['n_labels'] = 2000
manifest_dir = "manifests/commands/"
for tag in ['train', 'val', 'test']:
    si_config['{}_manifest'.format(tag)]=os.path.join(manifest_dir,'si_{}_{}_manifest.csv'.format("command", tag))

si_model = si_config['model_class'](si_config)
si_config['input_file'] = ""
si_config['output_file'] = "models/si_command_frames.pt"
print(si_model)
hk.train(si_config, model=si_model)