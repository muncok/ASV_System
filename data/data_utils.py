import os
import numpy as np
import pandas as pd

from .dataset import SpeechDataset, featDataset

def find_dataset(config):
    dataset_name = config ['dataset']
    if dataset_name == "voxc":
        config['data_folder'] = "dataset/voxceleb1/wav"
        config['input_dim'] = 64
        df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = SpeechDataset
    elif dataset_name == "voxc_mfcc":
        config['data_folder'] = "dataset/kaldi/voxceleb/feats/data/npy"
        config['input_dim'] = 23
        df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = featDataset
    elif dataset_name == "voxc_fbank":
        config['data_folder'] = \
        "dataset/kaldi/voxceleb/feats/data-fbank/fbank_npy"
        config['input_dim'] = 64
        df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = featDataset
    elif dataset_name == "voxc_fbank_xvector":
        config['data_folder'] = \
        "dataset/kaldi/voxceleb/feats/data-fbank/xvector_npy"
        config['input_dim'] = 64
        df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = featDataset
    elif dataset_name == "reddots":
        config['data_folder'] = "dataset/reddots_r2015q4_v1/wav"
        config['input_dim'] = 40
        df = pd.read_pickle(
                "dataset/dataframes/reddots/Reddots_Dataframe.pkl")
        n_labels = 70
        dset = SpeechDataset
    elif dataset_name == "gcommand_fbank_xvector":
        config['data_folder'] = \
        "dataset/kaldi/gcommand/feats/data-fbank/xvector_npy"
        config['input_dim'] = 64
        df = pd.read_pickle(
                "dataset/dataframes/gcommand/equal_num_102spk/equal_num_102spk_si.pkl")
        n_labels = 1759
        dset = featDataset

    if not os.path.isdir(config['data_folder']):
        print("there is no {} directory".format(config['data_folder']))
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
            print(len(val_df))
            if len(test_df) == 0:
                # in case of no explicit testset
                test_df = val_df
    else:
        print("split randomly")
        np.random.seed(3)
        test_df = df.sample(frac=0.2)
        train_df = df.drop(index=test_df.index)
        val_df = test_df.sample(frac=0.5)
        test_df = test_df.drop(index=val_df.index)

    return [train_df, val_df, test_df]


