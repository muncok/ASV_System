import torch
import torch.utils.data as data
# from .dataset import SpeechDataset

def _var_len_collate_fn(batch):
    """
    collate_fn with variable length sequencial data.
    it zero-padding short data
    :param batch:
    :return:
    """
    def func(p):
        return p[0].size(1)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(2)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, max_seqlength, freq_size)
    targets = []
    seq_lengths = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        seq_lengths.append(seq_length)
        inputs[x].narrow(1, 0, seq_length).copy_(tensor)
        targets.append(target)
    targets = torch.LongTensor(targets)
    return seq_lengths, inputs, targets

def init_default_loader(config, dataset, shuffle, var_len=False):
    '''
    Initialize the datasets, samplers and dataloaders for si training
    '''
    if not var_len:
        collate_fn=data.dataloader.default_collate
    else:
        collate_fn=_var_len_collate_fn
        
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=False,
            collate_fn=collate_fn,
            drop_last=False)
    return dataloader

def init_loaders(config, datasets, collate_fn=data.dataloader.default_collate):
    '''
    loaders from dataframes, train, val, test
    '''
    loaders = []
    for i, dataset in enumerate(datasets):
        if i == 0:
            loader = init_default_loader(config, dataset, shuffle=True, collate_fn=collate_fn)
        else:
            loader = init_default_loader(config, dataset, shuffle=False, collate_fn=collate_fn)
        loaders.append(loader)
    return loaders
