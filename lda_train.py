
import torch
import pickle
import numpy as np

from dnn_utils.sv_score import init_protonet, init_default_loaders, init_seed
from dnn_utils.utils.sv_parser import get_parser
from dnn_utils.sv_score import embeds
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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

