import numpy as np

from torch.utils.data.sampler import Sampler
                # sample_label = np.random.choice(self.labels_unique)

class HistSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(self.__len__()):
            labels_in_batch = set()
            inds = np.array([], dtype=np.int)

            while inds.shape[0] < self.batch_size:
                sample_label = np.random.choice(self.labels_unique)
                if sample_label in labels_in_batch:
                    continue

                labels_in_batch.add(sample_label)
                subsample_size = np.random.choice(range(5, 11))
                sample_label_ids = np.argwhere(np.in1d(self.labels, sample_label)).reshape(-1)
                subsample = np.random.permutation(sample_label_ids)[:subsample_size]
                inds = np.append(inds, subsample)

            inds = inds[:self.batch_size]
            yield list(inds)

    def __len__(self):
        return len(self.labels) // self.batch_size

class HistSampler_v1(Sampler):
    """
    make iterating over all classes for every epoch
    """
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.batch_size = batch_size

    def __iter__(self):
        # self.__len__ represents a number of batches in an epoch
        unused_labels = self.labels_unique.clone()
        np.random.shuffle(unused_labels)
        for i in range(self.__len__()):
            labels_in_batch = set()
            inds = np.array([], dtype=np.int)

            while inds.shape[0] < self.batch_size:
                # sample_label = np.random.choice(self.labels_unique)
                sample_label = unused_labels.pop()
                if sample_label in labels_in_batch:
                    continue

                labels_in_batch.add(sample_label)
                subsample_size = np.random.choice(range(5, 11))
                sample_label_ids = np.argwhere(np.in1d(self.labels, sample_label)).reshape(-1)
                subsample = np.random.permutation(sample_label_ids)[:subsample_size]
                inds = np.append(inds, subsample)

            inds = inds[:self.batch_size]
            yield list(inds)

    def __len__(self):
        return len(self.labels)*10 // self.batch_size

