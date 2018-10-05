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

def find_trial(config, basedir='./'):
    dataset_name = config ['dataset']
    if "voxc" in dataset_name:
        trial = pd.read_pickle(os.path.join(basedir,
            "dataset/dataframes/voxc1/voxc_trial.pkl"))
    elif "gcommand" in dataset_name:
        trial = pd.read_pickle(os.path.join(basedir,
            "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_trial.pkl"))
    else:
        print("No trial file")
        raise FileNotFoundError

    return trial

def find_dataset(config, basedir='./', split=True):
    dataset_name = config ['dataset']
    if dataset_name == "voxc1_wav":
        config['data_folder'] = "dataset/voxceleb1/voxceleb1_wav"
        config['input_dim'] = 64
        si_df = "dataset/dataframes/voxc1/si_voxc_dataframe.pkl"
        sv_df = "dataset/dataframes/voxc1/sv_voxc_dataframe.pkl"
        n_labels = 1211
        dset_class = SpeechDataset
    elif dataset_name == "voxc1_fbank_xvector":
        config['data_folder'] = "dataset/voxceleb1/feats/xvector_npy"
        config['input_dim'] = 64
        si_df = "dataset/dataframes/voxc1/si_voxc_dataframe.pkl"
        sv_df = "dataset/dataframes/voxc1/sv_voxc_dataframe.pkl"
        n_labels = 1211
        dset_class = featDataset
    elif dataset_name == "voxc12_fbank_xvector":
        config['data_folder'] = "dataset/voxceleb2/feats/xvector_npy"
        config['input_dim'] = 64
        si_df = "dataset/dataframes/voxc2/si_voxc12_dataframe.pkl"
        sv_df = "dataset/dataframes/voxc2/sv_voxc12_dataframe.pkl"
        n_labels = 7324
        dset_class = featDataset
    elif dataset_name == "reddots":
        config['data_folder'] = "dataset/reddots_r2015q4_v1/wav"
        config['input_dim'] = 40
        si_df = "dataset/dataframes/reddots/Reddots_Dataframe.pkl"
        n_labels = 70
        dset_class = SpeechDataset
    elif dataset_name == "gcommand_fbank_xvector":
        config['data_folder'] = "dataset/gcommand/feats/data-fbank/xvector_npy"
        config['input_dim'] = 64
        si_df = "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_si.pkl"
        sv_df = "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_sv1.pkl"
        n_labels = 1759
        dset_class = featDataset
    elif dataset_name == "gcommand_fbank":
        # no vad and cmvn
        config['data_folder'] = "dataset/gcommand/feats/data-fbank/fbank_npy"
        config['input_dim'] = 64
        si_df = "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_si.pkl"
        sv_df = "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_sv1.pkl"
        n_labels = 1759
        dset_class = featDataset
    elif dataset_name == "gcommand_equal30_wav":
        config['data_folder'] = "dataset/gcommand/gcommand_wav"
        config['input_dim'] = 64
        config["n_dct_filters"] = 64
        config["n_mels"] = 64
        si_df = "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_si.pkl"
        sv_df = "dataset/dataframes/gcommand/equal_num_30spk/equal_num_30spk_sv1.pkl"
        n_labels = 1759
        dset_class = SpeechDataset
    elif dataset_name == "voxc_11spks":
        config['data_folder'] = "dataset/voxceleb1/feats/xvector_npy"
        config['input_dim'] = 64
        config['fc_dims'] = 2
        si_df = "exp_embedding/si_voxc1_11spks.pkl"
        sv_df = "exp_embedding/sv_voxc1_11spks.pkl"
        n_labels = 11
        dset_class = featDataset
    elif dataset_name == "kor_voices":
        config['data_folder'] = "dataset/kor_commands/wav"
        config['input_dim'] = 64
        si_df = pd.read_pickle("dataset/kor_commands/kor_dataset.pkl")
        sv_df = si_df
        n_labels = 1759
        dset_class = SpeechDataset

    config['data_folder'] = os.path.join(basedir, config['data_folder'])
    if not 'dataset' in config or not os.path.isdir(config['data_folder']):
        print("there is no {} directory".format(config['data_folder']))
        raise FileNotFoundError

    # prefix the basedir
    si_df = pd.read_pickle(os.path.join(basedir, si_df))
    sv_df = pd.read_pickle(os.path.join(basedir, sv_df))

    config['n_labels'] = n_labels

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

