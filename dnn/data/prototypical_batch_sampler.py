# coding=utf-8
import numpy as np
import torch


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it', 'num_support', 'num_query',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations, randomize=False):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations:number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.randomize = randomize

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.idxs = range(len(self.labels))
        self.label_tens = dict()
        for idx, label in enumerate(self.labels):
            if label not in self.label_tens.keys():
                self.label_tens[label] = [idx]
            else:
                self.label_tens[label].append(idx)

        for k, v in self.label_tens.items():
            self.label_tens[k] = torch.LongTensor(v)

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        for it in range(self.iterations):
            cpi = np.random.randint(self.classes_per_it - 10, self.classes_per_it + 10) if self.randomize else self.classes_per_it
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            assert(len(self.classes) >= cpi)
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                assert(len(self.label_tens[c] >= spc))
                sample_idxs = torch.randperm(len(self.label_tens[c]))[:spc]
                batch[s] = self.label_tens[c][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


class wordBatchSampler(object):
    '''
    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, words, num_samples, iterations, randomize=False):
        '''
        - words = word ID, equal shape to labels
        - num_smpales = batch size
        - iterations:number of iterations (words) per epoch
        '''
        super(wordBatchSampler, self).__init__()
        self.words = words
        # self.classes_per_it = classes_per_it
        self.sample_per_batch = num_samples
        self.iterations = iterations

        self.randomize = randomize

        self.classes, self.counts = np.unique(self.words, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.idxs = range(len(self.words))
        self.word_tens = dict()
        for idx, word in enumerate(self.words):
            if word not in self.word_tens.keys():
                self.word_tens[word] = [idx]
            else:
                self.word_tens[word].append(idx)

        for k, v in self.word_tens.items():
            self.word_tens[k] = torch.LongTensor(v)

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spb = self.sample_per_batch
        for it in range(self.iterations):
            # cpi = np.random.randint(self.classes_per_it - 10, self.classes_per_it + 10) if self.randomize else self.classes_per_it
            wpi = 1  # word per iteration
            batch_size = spb * wpi
            batch = torch.LongTensor(batch_size)
            w_idxs = torch.randperm(len(self.words))[:wpi]
            # for i, c in enumerate(self.classes[w_idxs]):
            w = self.words[w_idxs]
            assert(len(self.word_tens[w] >= spb))
            sample_idxs = torch.randperm(len(self.word_tens[w]))[:spb]
            batch = self.word_tens[w][sample_idxs]
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations


