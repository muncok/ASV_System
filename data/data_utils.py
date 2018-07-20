import os
import numpy as np
import pandas as pd

from .dataset import SpeechDataset, featDataset

def find_dataset(config):
    dataset_name = config ['dataset']
    if dataset_name == "voxc":
        config['data_folder'] = "dataset/voxceleb/wav"
        config['input_dim'] = 40
        df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = SpeechDataset
    elif dataset_name == "voxc_mfcc":
        config['data_folder'] = "dataset/voxceleb/mfcc"
        config['input_dim'] = 40
        df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = featDataset
    elif dataset_name == "mini_voxc_mfcc":
        config['data_folder'] = "dataset/voxceleb/mfcc"
        config['input_dim'] = 20
        df = pd.read_pickle("dataset/dataframes/voxc/si_mini_voxc.pkl")
        n_labels = 70
        dset = featDataset
    elif dataset_name == "voxc_fbank":
        config['data_folder'] = "dataset/voxceleb/fbank"
        config['input_dim'] = 64
        df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = featDataset
    elif dataset_name == "voxc_fbank_xvector":
        config['data_folder'] = "dataset/voxceleb1/fbank-xvector"
        config['input_dim'] = 64
        df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = featDataset
    elif dataset_name == "voxc_fbank_xvector1":
        config['data_folder'] = \
        "dataset/kaldi/voxceleb/feats/data-fbank1/xvector_npy"
        config['input_dim'] = 64
        df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = featDataset
    elif dataset_name == "sess_voxc_mfcc":
        config['data_folder'] = "dataset/voxceleb/mfcc"
        config['input_dim'] = 20
        df = pd.read_pickle("dataset/dataframes/voxc/si_sess_voxc.pkl")
        n_labels = 215
        dset = featDataset
    elif dataset_name == "reddots":
        config['data_folder'] = "dataset/reddots_r2015q4_v1/wav"
        config['input_dim'] = 40
        df = pd.read_pickle(
                "dataset/dataframes/reddots/Reddots_Dataframe.pkl")
        n_labels = 70
        dset = SpeechDataset
    elif dataset_name == "reddots_vad":
        config['data_folder'] = "vad/reddots_vad/"
        config['input_dim'] = 40
        df = pd.read_pickle(
                "/home/muncok/DL/projects/sv_experiments/dataset/dataframes/reddots/reddots_vad.pkl")
        n_labels = 70
        dset = SpeechDataset

    if not os.path.isdir(config['data_folder']):
        raise FileNotFoundError

    config['n_labels'] = n_labels

    return df, dset

def split_df(df):
    if 'set' in df.columns:
        if df.set.dtype == 'int64':
            train_df = df[df.set == 1]
            val_df = df[df.set == 2]
            test_df = df[df.set == 3]
        else:
            train_df = df[(df.set == 'train')]
            val_df = df[(df.set == 'val')]
            test_df = df[(df.set == 'test')]
    else:
        print("split randomly")
        np.random.seed(3)
        test_df = df.sample(frac=0.2)
        train_df = df.drop(index=test_df.index)
        val_df = test_df.sample(frac=0.5)
        test_df = test_df.drop(index=val_df.index)

    return [train_df, val_df, test_df]


