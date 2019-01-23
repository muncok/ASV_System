import os
import pandas as pd
import warnings

from .speech_dataset import SpeechDataset
from .feat_dataset import FeatDataset

def split_df(df):
    if 'set' in df.columns:
        train_df = df[df.set == 1]
        test_df = df[df.set == 2]
    else:
        warnings.warn("No official splits")
        warnings.warn("Randomly splited to train:test = 8:2")
        test_df = df.sample(frac=0.2)
        train_df = df.drop(index=test_df.index)

    return [train_df, test_df]

def find_trial(config, basedir='./'):
    dataset = config['dataset']
    if "gcommand" in dataset:
        trial_name = "gcommand_equal_num_30spk_trial"
        trial = pd.read_csv(os.path.join(basedir,
            "/dataset/SV_sets/gcommand/equal_num_30spk/gcommand_sv_trial.csv"))
    elif "voxc1" in dataset:
        trial_name = "voxc1_sv_test"
        trial = pd.read_csv(os.path.join(basedir,
            "/dataset/SV_sets/voxceleb1/dataframes/voxc1_sv_trial.csv"))
    else:
        warnings.warn("ERROR: No trial file")
        raise FileNotFoundError
    print("=> Loaded trial: {}".format(trial_name))

    return trial

def get_dataset_info(config, dataset):
    try:
        name, in_format, in_dim, mode = dataset.split("_")
    except:
        warnings.warn("dataset format: {name}_{format}_{dim}_{wav|feat}")
    if mode == "wav":
        dataset_cls = SpeechDataset
        if name == "gcommand":
            config['data_folder'] = "/dataset/SV_sets/gcommand/wavs"
            config['input_format'] = in_format
            config['input_dim'] = int(in_dim)
            n_labels = 1759
            si_df = "/dataset/SV_sets/gcommand/equal_num_30spk/gcommand_si.csv"
            sv_df = "/dataset/SV_sets/gcommand/equal_num_30spk/gcommand_sv.csv"
        elif name == "voxc1":
            config['data_folder'] = "/dataset/SV_sets/voxceleb1/wavs"
            config['input_format'] = in_format
            config['input_dim'] = int(in_dim)
            n_labels = 1211
            si_df = "/dataset/SV_sets/voxceleb1/dataframes/voxc1_si.csv"
            sv_df = "/dataset/SV_sets/voxceleb1/dataframes/voxc1_sv.csv"
    elif mode == "feat":
        dataset_cls = FeatDataset
        if dataset == "voxc1_mfcc_30_feat":
            config['data_folder'] = "/dataset/SV_sets/voxceleb12/feats/mfcc30"
            config['input_format'] = in_format
            config['input_dim'] = int(in_dim)
            config['num_workers'] = 8
            n_labels = 7325
            si_df = "/dataset/SV_sets/voxceleb12/dataframes/voxc12_si.csv"
            sv_df = "/dataset/SV_sets/voxceleb12/dataframes/voxc12_sv.csv"
        elif dataset == "voxc2_fbank_64_feat":
            config['data_folder'] = "/dataset/SV_sets/voxceleb2/feats/fbank64_vad"
            config['input_format'] = in_format
            config['input_dim'] = int(in_dim)
            config['num_workers'] = 8
            n_labels = 6114
            si_df = "/dataset/SV_sets/voxceleb2/dataframes/voxc2_si.csv"
            sv_df = "/dataset/SV_sets/voxceleb2/dataframes/voxc2_sv.csv"
        elif dataset == "voxc12_fbank_64_feat":
            config['data_folder'] = "/dataset/SV_sets/voxceleb12/feats/fbank64_vad"
            config['input_format'] = in_format
            config['input_dim'] = int(in_dim)
            config['num_workers'] = 8
            n_labels = 7325
            si_df = "/dataset/SV_sets/voxceleb12/dataframes/voxc12_si.csv"
            sv_df = "/dataset/SV_sets/voxceleb12/dataframes/voxc12_sv.csv"

    if config['n_labels'] is None:
        config['n_labels'] = n_labels

    return dataset_cls, si_df, sv_df

def find_dataset(config, basedir='./', split=True):
    dataset = config['dataset']
    dataset_cls, si_df, sv_df = get_dataset_info(config, dataset)

    config['data_folder'] = os.path.join(basedir, config['data_folder'])
    if not 'dataset' in config or not os.path.isdir(config['data_folder']):
        print("Wrong directory {} ".format(config['data_folder']))
        raise FileNotFoundError

    if si_df.endswith(".csv"):
        si_df = pd.read_csv(os.path.join(basedir, si_df))
        sv_df = pd.read_csv(os.path.join(basedir, sv_df))
    elif si_df.endswith(".pkl"):
        si_df = pd.read_pickle(os.path.join(basedir, si_df))
        sv_df = pd.read_pickle(os.path.join(basedir, sv_df))

    # split dataframes
    if split: si_dfs = split_df(si_df)
    else: si_dfs = [si_df]

    # for computing eer, we need sv_df
    if not config["no_eer"]: dfs = si_dfs + [sv_df]
    else: dfs = si_dfs

    datasets = []
    for i, df in enumerate(dfs):
        if i == 0: datasets.append(dataset_cls.read_df(config, df, "train"))
        else: datasets.append(dataset_cls.read_df(config, df, "test"))

    return dfs, datasets
