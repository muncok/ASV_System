import os
import pandas as pd

from .dataset import SpeechDataset, featDataset

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
            assert len(val_df) != 0
            if len(test_df) == 0:
                # in case of no explicit testset
                test_df = val_df
    else:
        print("split dataset randomly")
        test_df = df.sample(frac=0.2)
        train_df = df.drop(index=test_df.index)
        val_df = test_df.sample(frac=0.5)
        test_df = test_df.drop(index=val_df.index)

    return [train_df, val_df, test_df]

def find_trial(config):
    dataset_name = config ['dataset']
    if "voxc" in dataset_name:
        trial = pd.read_pickle("dataset/dataframes/voxc1/voxc_trial.pkl")
    elif "gcommand" in dataset_name:
        trial = pd.read_pickle(
                "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_trial.pkl")
    else:
        print("No trial file")
        raise FileNotFoundError

    return trial

def find_dataset(config, split=True):
    dataset_name = config ['dataset']
    if dataset_name == "voxc":
        config['data_folder'] = "dataset/voxceleb1/wav"
        config['input_dim'] = 64
        si_df = pd.read_pickle("dataset/dataframes/voxc1/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset_class = SpeechDataset
    elif dataset_name == "voxc1_fbank":
        config['data_folder'] = \
        "dataset/kaldi/voxceleb/feats/data-fbank/xvector_npy"
        config['input_dim'] = 64
        si_df = pd.read_pickle("dataset/dataframes/voxc1/si_voxc_dataframe.pkl")
        sv_df = pd.read_pickle("dataset/dataframes/voxc1/sv_voxc_dataframe.pkl")
        n_labels = 1260
        dset_class = featDataset
    elif dataset_name == "voxc12_fbank":
        config['data_folder'] = \
        "dataset/voxceleb2/feats/xvector_npy"
        config['input_dim'] = 64
        si_df = pd.read_pickle("dataset/dataframes/voxc2/si_voxc12_dataframe.pkl")
        sv_df = pd.read_pickle("dataset/dataframes/voxc2/sv_voxc12_dataframe.pkl")
        n_labels = 7324
        dset_class = featDataset
    elif dataset_name == "voxc12_mfcc":
        config['data_folder'] = \
        "dataset/kaldi/voxceleb/xvector/data/train_combined_no_sil/xvector_npy"
        config['input_dim'] = 30
        si_df = pd.read_pickle("dataset/dataframes/voxc2/si_voxc12_dataframe.pkl")
        sv_df = pd.read_pickle("dataset/dataframes/voxc2/sv_voxc12_dataframe.pkl")
        n_labels = 7324
        dset_class = featDataset
    elif dataset_name == "reddots":
        config['data_folder'] = "dataset/reddots_r2015q4_v1/wav"
        config['input_dim'] = 40
        si_df = pd.read_pickle(
                "dataset/dataframes/reddots/Reddots_Dataframe.pkl")
        n_labels = 70
        dset_class = SpeechDataset
    elif dataset_name == "gcommand_fbank":
        config['data_folder'] = \
        "dataset/kaldi/gcommand/feats/data-fbank/xvector_npy"
        config['input_dim'] = 64
        si_df = pd.read_pickle(
                "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_si.pkl")
        sv_df = pd.read_pickle(
                "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_sv1.pkl")
        n_labels = 1759
        dset_class = featDataset
    elif dataset_name == "gcommand_fbank1":
        # no vad and cmvn
        config['data_folder'] = \
        "dataset/gcommand/feats/data-fbank/fbank_npy"
        config['input_dim'] = 64
        si_df = pd.read_pickle(
                "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_si.pkl")
        sv_df = pd.read_pickle(
                "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_sv1.pkl")
        n_labels = 1759
        dset_class = featDataset
    elif dataset_name == "gcommand_equal30_wav":
        config['data_folder'] = \
        "dataset/gcommand/gcommand_wav"
        config['input_dim'] = 64
        config["n_dct_filters"] = 64
        config["n_mels"] = 64
        si_df = pd.read_pickle(
                "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_si.pkl")
        sv_df = pd.read_pickle(
                "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_sv1.pkl")
        n_labels = 1759
        dset_class = SpeechDataset
    elif dataset_name == "gcommand_wav":
        config['data_folder'] = \
        "dataset/gcommand/gcommand_wav"
        config['input_dim'] = 40
        si_df = pd.read_pickle(
                "dataset/dataframes/gcommand/gcommand_dataframe.pkl")
        sv_df = si_df
        n_labels = 1759
        dset_class = SpeechDataset

    if not os.path.isdir(config['data_folder']):
        print("there is no {} directory".format(config['data_folder']))
        raise FileNotFoundError

    config['n_labels'] = n_labels

    if split:
        si_dfs = split_df(si_df)
        dfs = si_dfs + [sv_df]
    else:
        # si dataset is not splitted
        dfs = [si_df, sv_df]
    datasets = []
    for i, df in enumerate(dfs):
        if i == 0:
            datasets.append(dset_class.read_df(config, df, "train"))
        else:
            datasets.append(dset_class.read_df(config, df, "test"))

    return dfs, datasets

