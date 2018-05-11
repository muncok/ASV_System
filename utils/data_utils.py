def split_df(df):
    if 'set' in df.columns:
        train_df = df[df.set == 'train']
        val_df = df[df.set == 'val']
        test_df = df[df.set == 'test']
    else:
        raise("datafrmes does not have set attribute")

    return [train_df, val_df, test_df]


