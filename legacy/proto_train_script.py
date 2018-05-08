import torch
from dnn.train.prototypical_loss import prototypical_loss as p_loss
from dnn.train.model import init_protonet, init_seed
from dnn.prototypical_train import init_lr_scheduler, init_optim, train, evaluate
from dnn.data.dataloader import init_proto_loaders
from dnn.parser import get_sv_parser

def secToSample(sec):
    return int(16000 * sec)

options = get_sv_parser().parse_args()
options.train_manifest = "../interspeech2018/manifests/commands/words/" \
                         "sv_right_manifest.csv"
options.val_manifest = "../interspeech2018/manifests/commands/words/" \
                       "sv_right_manifest.csv"

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

options.splice_dim = 11
tr_dataloader, val_dataloader = init_proto_loaders(options)

#### train ####
print("training")
# options.input = "../interspeech2018/models/commands/word_aligned.pt"
model = init_protonet(options, small=True, fine_tune=False)
options.output = "../interspeech2018/models/commands/proto_right.pt"
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