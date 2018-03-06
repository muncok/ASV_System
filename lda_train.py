
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable

from dnn_utils.sv_score import init_protonet, init_default_loaders, init_seed
from dnn_utils.utils.sv_parser import get_parser
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def embeds(opt, val_dataloader, model):
    val_iter = iter(val_dataloader)
    nb_splicing = opt.input_length // opt.splice_length
    model.eval()
    embeddings = []
    labels = []
    for batch in tqdm(val_iter):
        x, y = batch
        time_dim = x.size(2)
        split_points = range(0, time_dim-(time_dim)//nb_splicing+1, time_dim//nb_splicing)
        model_outputs = []
        for point in split_points:
            x_in = Variable(x.narrow(2, point, time_dim//nb_splicing))
            if opt.cuda:
                x_in = x_in.cuda()
            model_outputs.append(model.embed(x_in))
        ## snippet scale LDA
        # model_output = torch.cat(model_outputs, dim=0)
        # y = torch.cat([y]*len(model_outputs), dim=0)
        ## uttrs scale LDA, it's better now
        model_output = torch.stack(model_outputs, dim=0)
        model_output = model_output.mean(0)
        embeddings.append(model_output.cpu().data.numpy())
        labels.append(y.numpy())
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels

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

input_uttr_sec = options.input_length
splice_l = options.splice_length
options.input_length = secToSample(input_uttr_sec)
options.splice_length = secToSample(splice_l)

tr_dataloader, val_dataloader = init_default_loaders(options)
model = init_protonet(options)
embedings, labels = embeds(options, tr_dataloader, model) # embeddings: sample x emb_size
n_samples = embedings.shape[0]

clf = LDA()
random_idx = np.random.permutation(np.arange(0,n_samples))
train_X, train_y = embedings[random_idx[:n_samples-100]], labels[random_idx[:n_samples-100]]
test_X, test_y = embedings[random_idx[-100:]], labels[random_idx[-100:]]
clf.fit(train_X, train_y)
score = clf.score(test_X, test_y)
print(score)
pickle.dump(clf, open(options.output, "wb"))

