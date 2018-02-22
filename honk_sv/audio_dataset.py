class audio_dataset():
    def __init__(self, name, data_path, df_path):
        self.name = name
        self.data_dir = data_path
        self.df = pd.read_pickle(df_path)
        self.all_spks = list(self.df.spk.unique())
        self.no_sents = True
        if hasattr(self.df, "sent"):
            self.no_sents = False
            self.all_sents = list(self.df.sent.unique())

    def random_audio(self):
        audio = self.df.sample(n=1)
        return audio
    
    def random_split(self, train_pct):
        # random sampling
        trainset = self.df.sample(frac=train_pct)
        testset = self.df.drop(index=si_random_train.index)
        valset = si_random_test.sample(frac=0.5)
        si_random_test = si_random_test.drop(index=si_random_val.index) 
        print("[random] train:{}, val:{}, test:{}".format(len(si_random_train),
                                                          len(si_random_val), 
                                                          len(si_random_test)))
        return si_random_train, si_r
    
    def audio_by_spk(self, spk_names):
        if not isinstance(spk_names, list):
            spk_names = [spk_names]
        return self.df[self.df.spk.isin(spk_names)]
    
    def audio_by_sent(self, sents):
        if not isinstance(sents, list):
            sents = [sents]
        return self.df[self.df.sent.isin(sents)]    
    
    def write_manifest(self, dataset, save_path, shuffle=True):
        if not isinstance(dataset, pd.DataFrame):
            dataset = self.df[dataset]
            
        with open(save_path, 'w') as f:
            samples = []
            for index, row in dataset.iterrows():
                if self.name == 'command':
                    file_path = os.path.join(data_dir, row.sent, row.file)
                else:
                    file_path = os.path.join(self.data_dir, row.spk, row.file)
                label = self.all_spks.index(row.spk)
                sample = ','.join([file_path, str(label)])
                samples.append(sample)
            if shuffle:
                random.shuffle(samples)
            writer = csv.writer(f, delimiter='\n', quoting=csv.QUOTE_NONE)
            writer.writerow(samples)
            print("{} was written".format(save_path))