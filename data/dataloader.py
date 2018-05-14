import torch
import torch.utils.data as data

from .prototypical_batch_sampler import PrototypicalBatchSampler
from .verification_batch_sampler import VerificationBatchSampler

def _collate_fn(batch):
    """
    collate_fn with variable length sequencial data.
    it zero-padding short data
    :param batch:
    :return:
    """
    def func(p):
        return p[0].size(0)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(1)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(0)
    inputs = torch.zeros(minibatch_size, 1, max_seqlength,freq_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        inputs[x][0].narrow(0, 0, seq_length).copy_(tensor)
        targets.append(target)
    targets = torch.LongTensor(targets)
    return inputs, targets

def init_default_loader(config, dataset, shuffle):
    '''
    Initialize the datasets, samplers and dataloaders for si training
    '''
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    val_dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=shuffle)
    return val_dataloader

def init_maxlengh_loader(dataset):
    '''
    Initialize the datasets, samplers and dataloaders for si training
    '''
    val_dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=64,
                                                 num_workers=8,
                                                 collate_fn=_collate_fn)
    return val_dataloader

def init_prototypical_loaders(opt, train_dataset, val_dataset):
    '''
    dataloader with PrototypicalBatchSampler
    '''
    tr_sampler = PrototypicalBatchSampler(labels=train_dataset.audio_labels,
                                          classes_per_it=opt.classes_per_it_tr,
                                          num_samples=opt.num_support_tr + opt.num_query_tr,
                                          iterations=opt.iterations,
                                          randomize=False)

    val_sampler = PrototypicalBatchSampler(labels=val_dataset.audio_labels,
                                           classes_per_it=opt.classes_per_it_val,
                                           num_samples=opt.num_support_val + opt.num_query_val,
                                           iterations=opt.iterations,
                                           randomize=False)


    tr_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_sampler=tr_sampler,
                                                num_workers=16,
                                                collate_fn=_collate_fn)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_sampler=val_sampler,
                                                 num_workers=8,
                                                 collate_fn=_collate_fn)
    return tr_dataloader, val_dataloader

def init_verification_loader(opt, dataset):
    '''
    dataloader with VerificationBatchSampler
    '''
    val_sampler = VerificationBatchSampler(labels=dataset.audio_labels,
                                           classes_per_it=opt.classes_per_it_val,
                                           num_support=opt.num_support_val,
                                           num_query=opt.num_query_val,
                                           iterations=opt.iterations)

    val_dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_sampler=val_sampler,
                                                 num_workers=8,
                                                 collate_fn=_collate_fn)
    return val_dataloader

