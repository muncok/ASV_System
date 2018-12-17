import os
import pandas as pd

from .dataset import SpeechDataset

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
    dataset = config ['dataset']
    if "gcommand" in dataset:
        trial_name = "gcommand_equal_num_30spk_trial"
        trial = pd.read_pickle(os.path.join(basedir,
            "dataset/gcommand/equal_num_30spk_test_trial.pkl"))
    else:
        print("ERROR: No trial file")
        raise FileNotFoundError

    print("=> loaded trial: {}".format(trial_name))

    return trial

def find_dataset(config, basedir='./'):
    dataset = config ['dataset']
    config['data_folder'] = "asdf"
    if dataset == "gcommand_fbank40":
        config['data_folder'] = "dataset/gcommand/wav"
        config['input_dim'] = 40
        config['input_format'] = 'fbank'
        si_df = "dataset/gcommand/equal_num_30spk_si.pkl"
        sv_df = "dataset/gcommand/equal_num_30spk_sv.pkl"
        n_labels = 1759
    elif dataset == "gcommand_mfcc40":
        config['data_folder'] = "dataset/gcommand/wav"
        config['input_dim'] = 40
        config['input_format'] = 'mfcc'
        si_df = "dataset/gcommand/equal_num_30spk_si.pkl"
        sv_df = "dataset/gcommand/equal_num_30spk_sv.pkl"
        n_labels = 1759

    config['data_folder'] = os.path.join(basedir, config['data_folder'])
    if not 'dataset' in config or not os.path.isdir(config['data_folder']):
        print("Wrong directory {} ".format(config['data_folder']))
        raise FileNotFoundError

    if config['n_labels'] == None:
        config['n_labels'] = n_labels

    si_df = pd.read_pickle(os.path.join(basedir, si_df))
    sv_df = pd.read_pickle(os.path.join(basedir, sv_df))

    # splitting dataframes
    si_dfs = split_df(si_df)

    # for computing eer, we need sv_df
    if not config["no_eer"]:
        dfs = si_dfs + [sv_df]
    else:
        dfs = si_dfs

    datasets = []
    for i, df in enumerate(dfs):
        if i == 0:
            datasets.append(SpeechDataset.read_df(config, df, "train"))
        else:
            datasets.append(SpeechDataset.read_df(config, df, "test"))

    return dfs, datasets
