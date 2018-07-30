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


def find_dataset(config):
    dataset_name = config ['dataset']
    if dataset_name == "voxc":
        config['data_folder'] = "dataset/voxceleb1/wav"
        config['input_dim'] = 64
<<<<<<< HEAD
        df = pd.read_pickle("dataset/voxceleb1/dataframe/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = SpeechDataset
    elif dataset_name == "voxc_mfcc":
        config['data_folder'] = "dataset/kaldi/voxceleb/feats/data/npy"
        config['input_dim'] = 23
        df = pd.read_pickle("dataset/voxceleb1/dataframe/si_voxc_dataframe.pkl")
=======
        si_df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
>>>>>>> d662b56b9ae999b501d931653c8cb430499f733a
        n_labels = 1260
        dset_class = SpeechDataset
    elif dataset_name == "voxc1_fbank":
        config['data_folder'] = \
        "dataset/kaldi/voxceleb/feats/data-fbank/xvector_npy"
        config['input_dim'] = 64
<<<<<<< HEAD
        df = pd.read_pickle("dataset/voxceleb1/dataframe/si_voxc_dataframe.pkl")
=======
        si_df = pd.read_pickle("dataset/dataframes/voxc/si_voxc_dataframe.pkl")
        sv_df = pd.read_pickle("dataset/dataframes/voxc/sv_voxc_dataframe.pkl")
>>>>>>> d662b56b9ae999b501d931653c8cb430499f733a
        n_labels = 1260
        dset_class = featDataset
    elif dataset_name == "voxc12_fbank":
        config['data_folder'] = \
        "dataset/kaldi/voxceleb/xvector/data/data-fbank/xvector_npy"
        config['input_dim'] = 64
<<<<<<< HEAD
        df = pd.read_pickle("dataset/voxceleb1/dataframe/si_voxc_dataframe.pkl")
        n_labels = 1260
        dset = featDataset
    elif dataset_name == "voxc12_fbank_xvector":
        config['data_folder'] = \
        "dataset/kaldi/voxceleb/xvector/data/data-fbank/xvector_npy"
        config['input_dim'] = 64
        df = pd.read_pickle("dataset/voxceleb2/dataframe/si_voxc12_dataframe.pkl")
        n_labels = 7324
        # df = pd.read_pickle("dataset/voxceleb1/dataframe/si_voxc_dataframe.pkl")
        # n_labels = 1260
        dset = featDataset
=======
        si_df = pd.read_pickle("dataset/voxceleb2/dataframe/si_voxc12_dataframe.pkl")
        sv_df = pd.read_pickle("dataset/voxceleb2/dataframe/sv_voxc12_dataframe.pkl")
        n_labels = 7324
        dset_class = featDataset
>>>>>>> d662b56b9ae999b501d931653c8cb430499f733a
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
<<<<<<< HEAD
        df = pd.read_pickle(
                "dataset/gcommand/dataframe/equal_num_30spk/equal_num_30spk_si.pkl")
=======
        si_df = pd.read_pickle(
                "dataset/dataframes/gcommand/equal_num_102spk/equal_num_102spk_si.pkl")
>>>>>>> d662b56b9ae999b501d931653c8cb430499f733a
        n_labels = 1759
        dset_class = featDataset

    if not os.path.isdir(config['data_folder']):
        print("there is no {} directory".format(config['data_folder']))
        raise FileNotFoundError

    config['n_labels'] = n_labels

    si_dfs = split_df(si_df)
    dfs = si_dfs + [sv_df]
    datasets = []
    for i, df in enumerate(dfs):
        if i == 0:
            datasets.append(dset_class.read_df(config, df, "train"))
        else:
<<<<<<< HEAD
            train_df = df[(df.set == 'train')]
            val_df = df[(df.set == 'val')]
            test_df = df[(df.set == 'test')]
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
=======
            datasets.append(dset_class.read_df(config, df, "test"))
>>>>>>> d662b56b9ae999b501d931653c8cb430499f733a

    return datasets

