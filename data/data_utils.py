import os
import pandas as pd

from .dataset import featDataset

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

def find_trial(config, basedir='./'):
    dataset_name = config ['dataset']
    if "voxc" in dataset_name:
        trial_name = "voxc12_test_trial"
        trial = pd.read_pickle(os.path.join(basedir,
            "dataset/voxceleb12/dataframes/voxc12_test_trial.pkl"))
    elif "gcommand" in dataset_name:
        trial_name = "gcommand_equal_num_30spk_trial"
        trial = pd.read_pickle(os.path.join(basedir,
            "dataset/gcommand/dataframes/equal_num_30spk/equal_num_30spk_trial.pkl"))
    else:
        print("ERROR: No trial file")
        raise FileNotFoundError

    print("=> loaded trial: {}".format(trial_name))

    return trial

def find_dataset(config, basedir='./', split=True):
    dataset_name = config ['dataset']
    if dataset_name == "voxc12_mfcc30":
        config['data_folder'] = "dataset/voxceleb12/feats/mfcc30"
        config['input_dim'] = 30
        config['input_format'] = 'mfcc'
        si_df = "dataset/voxceleb12/dataframes/voxc12_si_train_dataframe.pkl"
        sv_df = "dataset/voxceleb12/dataframes/voxc12_sv_test_dataframe.pkl"
        n_labels = 7325
        dset_class = featDataset
    elif dataset_name == "voxc1_mfcc30":
        config['data_folder'] = "dataset/voxceleb12/feats/mfcc30"
        config['input_dim'] = 30
        config['input_format'] = 'mfcc'
        si_df = "dataset/voxceleb12/dataframes/voxc1_si_train_dataframe.pkl"
        sv_df = "dataset/voxceleb12/dataframes/voxc12_sv_test_dataframe.pkl"
        n_labels = 1211
        dset_class = featDataset
    elif dataset_name == "voxc12_fbank64_vad":
        config['data_folder'] = "dataset/voxceleb12/feats/fbank64_vad"
        config['input_dim'] = 64
        config['input_format'] = 'fbank'
        si_df = "dataset/voxceleb12/dataframes/voxc12_si_train_dataframe.pkl"
        sv_df = "dataset/voxceleb12/dataframes/voxc12_sv_test_dataframe.pkl"
        n_labels = 7325
        dset_class = featDataset
    elif dataset_name == "voxc1_fbank64_vad":
        config['data_folder'] = "dataset/voxceleb12/feats/fbank64_vad"
        config['input_dim'] = 64
        config['input_format'] = 'fbank'
        si_df = "dataset/voxceleb12/dataframes/voxc1_si_train_dataframe.pkl"
        sv_df = "dataset/voxceleb12/dataframes/voxc12_sv_test_dataframe.pkl"
        n_labels = 1211
        dset_class = featDataset
    elif dataset_name == "gcommand_fbank64_vad":
        config['data_folder'] = "dataset/gcommand/feats/fbank64_vad"
        config['input_dim'] = 64
        config['input_format'] = 'fbank'
        si_df = "dataset/gcommand/dataframes/equal_num_30spk/equal_num_30spk_si.pkl"
        sv_df = "dataset/gcommand/dataframes/equal_num_30spk/equal_num_30spk_sv1.pkl"
        n_labels = 1759
        dset_class = featDataset
    elif dataset_name == "gcommand_mfcc30":
        config['data_folder'] = "dataset/gcommand/feats/mfcc30"
        config['input_dim'] = 30
        config['input_format'] = 'mfcc'
        si_df = "dataset/gcommand/dataframes/equal_num_30spk/equal_num_30spk_si.pkl"
        sv_df = "dataset/gcommand/dataframes/equal_num_30spk/equal_num_30spk_sv1.pkl"
        n_labels = 1759
        dset_class = featDataset

    config['data_folder'] = os.path.join(basedir, config['data_folder'])
    if not 'dataset' in config or not os.path.isdir(config['data_folder']):
        print("there is no {} directory".format(config['data_folder']))
        raise FileNotFoundError

    if config['n_labels'] == None:
        config['n_labels'] = n_labels

    # prefix the basedir
    si_df = pd.read_pickle(os.path.join(basedir, si_df))
    sv_df = pd.read_pickle(os.path.join(basedir, sv_df))

    if split:
        si_dfs = split_df(si_df)
    else:
        # si dataset is not splitted
        si_dfs = [si_df]

    if not config["no_eer"]:
        dfs = si_dfs + [sv_df]
    else:
        dfs = si_dfs

    datasets = []
    for i, df in enumerate(dfs):
        if i == 0:
            datasets.append(dset_class.read_df(config, df, "train"))
        else:
            datasets.append(dset_class.read_df(config, df, "test"))

    return dfs, datasets

