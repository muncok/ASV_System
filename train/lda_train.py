import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class LDAModel():
    def __init__(self):
        self.classifier = LDA()

    def fit(self, embeddings, labels):
        n_samples = embeddings.shape[0]
        n_test = 500 # for test samples
        random_idx = np.random.permutation(np.arange(0,n_samples))
        train_X, train_y = embeddings[random_idx[:n_samples-n_test]], \
            labels[random_idx[:n_samples-n_test]]
        test_X, test_y = embeddings[random_idx[-n_test:]], \
            labels[random_idx[-n_test:]]
        self.classifier.fit(train_X, train_y)
        score = self.classifier.score(test_X, test_y)
        print(score) # test_score

# LDA
# Reddots LDA
# reddots_files = pd.read_pickle("../dataset/dataframes/reddots/Reddots_Dataframe.pkl").file
# used_files = pd.concat([trn.file, ndx_file.file])
# unused_files = reddots_files[~reddots_files.isin(used_files)]

# lda_file = pd.DataFrame(unused_files, columns=['file'])
# lda_dataset = SpeechDataset.read_df(si_config, lda_file, "test")
# val_dataloader = init_default_loader(si_config, lda_dataset, shuffle=False)
# lda_embeddings, _ = embeds_utterance(si_config, val_dataloader, model, lda)

