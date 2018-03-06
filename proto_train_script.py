import torch
from dnn_utils.train.prototypical_loss import prototypical_loss as p_loss
from dnn_utils.sv_score import init_protonet, init_proto_loaders, init_seed, init_lr_scheduler, init_optim
from dnn_utils.sv_score import train, evaluate
from dnn_utils.utils.sv_parser import get_parser

def secToSample(sec):
    return int(16000 * sec)

options = get_parser().parse_args()
options.train_manifest = "manifests/reddots/si_reddots_train_manifest.csv"
options.val_manifest = "manifests/reddots/si_reddots_train_manifest.csv"

# audio options
options.n_dct_filters = 40
options.n_mels = 40
options.timeshift_ms = 100
options.data_folder = "/home/muncok/DL/dataset/SV_sets"
options.window_size= 0.025
options.window_stride= 0.010
options.cache_size = 32768
options.input_format = "mfcc"

if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

init_seed(options)

options.input_length = secToSample(options.input_length)
tr_dataloader, val_dataloader = init_proto_loaders(options)

#### train ####
print("training")
model = init_protonet(options)
optim = init_optim(options, model)
lr_scheduler = init_lr_scheduler(options, optim)
train(opt=options,
      tr_dataloader=tr_dataloader,
      val_dataloader=val_dataloader,
      model=model,
      optim=optim,
      lr_scheduler=lr_scheduler,
      loss=p_loss)

#### eval ####
print("evaluating")
model = init_protonet(options)
evaluate(options, val_dataloader, model, loss=p_loss)