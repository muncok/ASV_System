# coding=utf-8
import numpy as np
import itertools


class VerificationBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'nb_enroll_spks', 'num_support', 'nb_test_uttrs',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'nb_test_uttrs' samples
    for 'nb_enroll_spks' random classes.

    __len__ returns the number of episodes per epoch (same as 'iterations').
    '''

    def __init__(self, labels, classes_per_it, num_support, num_query, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - nb_enroll_spks: number of random classes for each iteration
        - num_support: number of support samples for each iteration for each class
        - nb_test_uttrs: number of query samples for each iteration for each class
        - iterations:number of iterations (episodes) per epoch
        '''
        super(VerificationBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_support + num_query  # n_support + n_pos_query
        self.num_support = num_support
        self.num_query = num_query  # n_pos_query == n_neg_query
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.idxs = range(len(self.labels))
        self.ndclasses = dict()
        for idx, label in enumerate(self.labels):
            if label not in self.ndclasses.keys():
                self.ndclasses[label] = [idx]
            else:
                self.ndclasses[label].append(idx)

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        for it in range(self.iterations):
            batch = np.zeros(
                ((self.sample_per_class + self.num_query) * self.classes_per_it), dtype=int)
            np.random.shuffle(self.classes)
            curr_classes = self.classes[:self.classes_per_it]
            not_curr_classes = self.classes[self.classes_per_it:]
            for i, c in enumerate(curr_classes):
                s = slice(i * self.sample_per_class,
                          (i + 1) * self.sample_per_class)
                pos_sample = np.random.choice(
                    self.ndclasses[c], self.sample_per_class, replace=False)
                batch[s] = pos_sample
            total_neg_samples = [v for k, v in self.ndclasses.items() if k in not_curr_classes]
            total_neg_samples = list(itertools.chain.from_iterable(total_neg_samples))
            neg_sample = np.random.choice(
                total_neg_samples,self.classes_per_it*self.num_query,
                replace=False)
            s = slice(self.sample_per_class * self.classes_per_it, len(batch))
            batch[s] = neg_sample
            # np.random.shuffle(batch) # this should be uselsess (?)
            yield batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
