# coding=utf-8
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.nn.modules.loss import _assert_no_grad


class PrototypicalLoss(Module):
    '''
    Class à la PyTorch for the prototypical loss function defined below
    '''

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        _assert_no_grad(target)
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cosine_similarity(x, y, filter_type=None):
    '''
    Compute euclidean distance between two tensors
    '''

    # x: N x D x K
    # y: M x D x 1
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    k = x.size(2)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d, k)
    y = y.unsqueeze(0).unsqueeze(3).expand(n, m, d, k)
    if filter_type == "diff":
        score = F.cosine_similarity(x, y, dim=2)  # (n,m,k)
        score = score.transpose_(1,2)  # (n,k,m)
        score_top2, top2_idxs = torch.topk(score, 2, dim=2, largest=True)  # (n,k,2)
        score_top2_diff =  score_top2[:,:,0] - score_top2[:,:,1]  # (n,k)
        _, topk_diff_idxs = torch.topk(score_top2_diff, k//2, dim=1) # (n, topk)
        topk_diff_idxs = topk_diff_idxs.unsqueeze(1).unsqueeze(2).expand(n,m,d,-1)
        return F.cosine_similarity(x.gather(3, topk_diff_idxs).mean(-1), y.mean(-1), dim=2)
    else:
        return F.cosine_similarity(x.mean(-1), y.mean(-1), dim=2)

def prototypical_loss(input, target, n_support, randomize=False, filter=None):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    cputargs = target.cpu() if target.is_cuda else target
    cputargs = cputargs.data
    cpuinput = input.cpu() if target.is_cuda else input

    if randomize and n_support > 1:
        n_support = np.random.randint(1, n_support)

    def supp_idxs(c):
        return torch.nonzero(cputargs.eq(int(c)))[:n_support].squeeze()

    classes = np.unique(cputargs)
    n_classes = len(classes)
    n_query = len(torch.nonzero(cputargs.eq(int(classes[0])))) - n_support

    os_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([cpuinput[i].mean(0).mean(-1) for i in os_idxs])

    prototypes = prototypes.cuda() if target.is_cuda else prototypes
    oq_idxs_0 = torch.stack(list(map(lambda c: torch.nonzero(cputargs.eq(int(c)))[n_support:], classes))).view(-1)
    oq_idxs_0 = oq_idxs_0.cuda() if target.is_cuda else oq_idxs_0
    oq = input[oq_idxs_0]
    # dists = euclidean_dist(oq, prototypes)
    dists = -cosine_similarity(oq, prototypes,filter)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    target_inds = target_inds.long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda() if target.is_cuda else target_inds

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)

    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    return loss_val,  acc_val

