import torch
import torch.nn as nn

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    # torch.cuda.cudnn_enabled = False
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def num_flat_features(x):
    '''
    flatten the activations in the way conv to fc
    :param x:
    :return:
    '''
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

def truncated_normal(tensor, std_dev=0.01):
    tensor.zero_()
    tensor.normal_(std=std_dev)
    while torch.sum(torch.abs(tensor) > 2 * std_dev) > 0:
        t = tensor[torch.abs(tensor) > 2 * std_dev]
        t.zero_()
        tensor[torch.abs(tensor) > 2 * std_dev] = torch.normal(t, std=std_dev)

class SerializableModule(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        print("saved to {}".format(filename))

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        print("loaded from {}".format(filename))

    def load_partial(self, filename):
        to_state = self.state_dict()
        from_state = torch.load(filename)
        valid_state = {k:v for k,v in from_state.items() if k in to_state.keys()}
        valid_state.pop('output.weight', None)
        valid_state.pop('output.bias', None)
        to_state.update(valid_state)
        self.load_state_dict(to_state)
        assert(len(valid_state) > 0)
        print("loaded from {}".format(filename))

class statistic_pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (batch, 1, time, bank)
        mean = x.mean(1, keep_dims=False)
        std = x.std(1, keep_dims=False)

