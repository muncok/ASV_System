
import numpy as np
import torch
import torch.utils.data as data
# import torch.nn.functional as F

from . import dataset as dset

splice_length_ = 50

def get_loader(config, datasets=None, _collate_fn=data.dataloader.default_collate):
    num_workers_ = config["num_workers"]
    batch_size = config['batch_size']
    if datasets is None:
        train_set, dev_set, test_set = dset.SpeechDataset.read_manifest(config)
    else:
        train_set, dev_set, test_set = datasets

    collate_fn_ = _collate_fn

    if train_set is not None:
        train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                       shuffle=True, drop_last=True, num_workers=num_workers_,
                                       collate_fn=collate_fn_)
    if dev_set is not None:
        dev_loader = data.DataLoader(dev_set, batch_size=min(len(dev_set), batch_size), shuffle=True,
                                     num_workers=num_workers_, collate_fn=collate_fn_)
    test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), batch_size), shuffle=True,
                                  num_workers=num_workers_, collate_fn=collate_fn_)
    return train_loader, dev_loader, test_loader

def _random_frames_collate_fn(batch):
    splice_length = splice_length_
    half_splice = splice_length //2
    minibatch_size = len(batch)
    tensors = []
    targets = []
    # inputs = torch.zeros(minibatch_size, splice_length, 40)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        # random
        point = np.random.randint(half_splice, tensor.size(0)-half_splice+1)
        tensor = tensor[point-half_splice:point+half_splice]
        tensors.append(tensor)
        targets.append(target)
        ## no overlap
        # nb_spliced = tensor.size(0) // splice_length
        # tensors.extend(tensor.split(splice_length, 0)[:-1])  # abandon residual
        # targets.extend(target*(nb_spliced))
    inputs = torch.stack(tensors)
    targets = torch.LongTensor(targets)
    return inputs, targets

def _nooverlap_frames_collate_fn(batch):
    minibatch_size = len(batch)
    tensors = []
    targets = []
    # inputs = torch.zeros(minibatch_size, splice_length, 40)
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        # no overlap
        nb_spliced = tensor.size(0) // splice_length_
        tensors.extend(tensor.split(splice_length_, 0)[:-1])  # abandon residual
        targets.extend([target]*(nb_spliced))
    inputs = torch.stack(tensors)
    targets = torch.LongTensor(targets)
    return inputs, targets

def _all_frames_collate_fn(batch):
    half_splice = splice_length_ //2
    minibatch_size = len(batch)
    tensors = []
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = torch.zeros(splice_length_, 40)
        tensor = sample[0]
        target = sample[1]

        # all frames
        points = np.arange(half_splice, tensor.size(0) - half_splice)
        for point in points:
            tensors.append(tensor[point-half_splice:point+half_splice])
            targets.append(target)
    inputs = torch.stack(tensors)
    targets = torch.LongTensor(targets)
    return inputs, targets

def _mtl_collate_fn(batch):
    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, batch[0][0].size(0), batch[0][0].size(1))
    targets = []
    targets1 = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = int(sample[1])
        target1 = int(sample[2])
        # target = [x for x in sample[1:]]
        inputs[x].copy_(tensor)
        targets.append(target)
        targets1.append(target1)
    targets, targets1 = torch.LongTensor(targets), torch.LongTensor(targets1)
    return inputs, targets, targets1
