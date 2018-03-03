import csv
import itertools
import torch
from protonet.speech_train import sv_score, sv_optimal_score, init_protonet, init_speechnet,\
    init_sv_loaders, init_seed
from protonet.parser import get_parser


def secToSample(sec):
    return int(16000 * sec)

options = get_parser().parse_args()
options.train_manifest = "manifests/reddots/si_reddots_train_manifest.csv"
options.val_manifest = "manifests/reddots/si_reddots_train_manifest.csv"
# options.val_manifest = "manifests/reddots/sv_reddots_manifest.csv"
# options.train_manifest = "manifests/voxc/fewshot/si_voxc_train_manifest.csv"
# options.val_manifest = "manifests/voxc/fewshot/si_voxc_val_manifest.csv"
options.n_dct_filters = 40
options.n_mels = 40
options.timeshift_ms = 100
options.data_folder = "/home/muncok/DL/dataset/SV_sets"
options.window_size= 0.025
options.window_stride= 0.010
options.cache_size = 32768

if torch.cuda.is_available() and not options.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

init_seed(options)

input_uttr_sec = 3
options.input_length = secToSample(input_uttr_sec)
options.num_query_val = 10
enroll_length = 30
options.iterations = 100

# sweep conditions
options.num_support_val = int(enroll_length/input_uttr_sec)
# options.input = "models/voxc/si_train/si_voxc_frames_res8w_1.pt"
options.input = "models/reddots/si_reddots_0.2s_random.pt"
model = init_protonet(options)
# model = init_speechnet(options)
test_uttrs_lens = [0.2, 0.5, 1, 2, 3]
splice_lens = [0.2, 0.5, 1, 2, 3]
conditions = itertools.product(test_uttrs_lens, splice_lens)
results = []
for test_uttrs_l, splice_l in conditions:
    if splice_l > test_uttrs_l: continue
    # fixing the enrollment speech duration
    options.num_test_frames = int(test_uttrs_l/splice_l)
    options.splice_length = secToSample(splice_l)
    _, val_dataloader = init_sv_loaders(options)
    full_eer = sv_optimal_score(options, val_dataloader, model, "full") # full
    random_eer = sv_optimal_score(options, val_dataloader, model, "random")
    diff_eer = sv_optimal_score(options, val_dataloader, model, "diff")
    results.append("{:2.1f}, {:2.1f}, {:2.2f}, {:2.2f}, {:2.2f}".format(test_uttrs_l, splice_l,
                                                   full_eer[0], random_eer[0], diff_eer[0]))

with open("results/sv_results_voxc_full.csv", "w") as f:
     f.write(options.input+"\n")
     f.write(options.val_manifest+"\n")
     f.write("enroll_sec:{}, cVa:{}, nsVa:{}, nqVa:{}\n\n".format(
         enroll_length,
         options.classes_per_it_val,
         options.num_support_val,
         options.num_query_val)
     )
     f.write("uttrs_l, splice_l, full_eer, random_eer, diff_eer\n")
     writer = csv.writer(f, delimiter='\n', quoting=csv.QUOTE_NONE)
     writer.writerow(results)