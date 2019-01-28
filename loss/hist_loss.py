import torch
import ipdb

from numpy.testing import assert_almost_equal

eps = 1e-5

class HistogramLoss(torch.nn.Module):
    def __init__(self, num_steps, cuda=True):
        super(HistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.cuda = cuda
        self.t = torch.arange(-1, 1+self.step, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        if self.cuda:
            self.t = self.t.cuda()

    def forward(self, features, classes):
        def histogram(inds, size):
            """
            :param inds: pos_inds or neg_inds
            :param size: n_pos_dist
            :return: histogram, weight(delta)
            """
            # s_repeat: (nbins, n_valid_dists)
            s_repeat_ = s_repeat.clone()
            # computing weights(delta)
            # indsa: check left boundary, (self.t - self.step) represents left boundaries
            indsa = (s_repeat_floor - (self.t - self.step) > -eps) & (s_repeat_floor - (self.t - self.step) < eps) & inds
            # number of nonzeros must be equal to pos_size or neg_size
            # it means each dist must be mapped to one bin, each column has only single one
            if indsa.nonzero().size()[0] != size:
                ipdb.set_trace()
            assert indsa.nonzero().size()[0] == size, (f'eps is inadequate, {indsa.nonzero().size()[0]}, {size}')
            zeros = torch.zeros((1, indsa.size()[1])).byte()
            if self.cuda:
                zeros = zeros.cuda()
            # indsb: check right boundary
            indsb = torch.cat((indsa, zeros))[1:, :]
            s_repeat_[~(indsb|indsa)] = 0
            s_repeat_[indsa] = (s_repeat_-(self.t-self.step))[indsa] / self.step
            s_repeat_[indsb] =  ((self.t + self.step)-s_repeat_)[indsb] / self.step
            # accumulates weights for each bin and normalizes them to be PMF
            return s_repeat_.sum(1) / size
        # features: (batch, 512), classes: (batch,)
        features, classes = features, classes
        classes_size = classes.size()[0]
        # classes_eq: (batch, batch)
        classes_eq = (classes.repeat(classes_size, 1)  == classes.view(-1, 1).repeat(1, classes_size)).data
        # dist(ance): (batch, batch)
        dists = torch.mm(features, features.transpose(0, 1))
        # s_inds: upper triangle one matrix, (batch, batcy), it filters valid dists
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).byte()
        if self.cuda:
            s_inds= s_inds.cuda()
        # pos_inds: 1D array, it is repeated self.tsize times, tsize==nbins
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        # pos_size: number of positive pairs
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()
        # s: flattend valid dists
        s = dists[s_inds].view(1, -1)
        # s_repeat: (nbins, n_valid_dists)
        s_repeat = s.repeat(self.tsize, 1)
        # s: quantize dists to step size.
        s_repeat_floor = (torch.floor(s_repeat.data / self.step) * self.step).float()

        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(histogram_pos.sum().item(), 1, decimal=2,
                            err_msg='Not good positive histogram', verbose=True)
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(histogram_neg.sum().item(), 1, decimal=2,
                            err_msg='Not good negative histogram', verbose=True)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1, histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.size()), -1).byte()
        if self.cuda:
            histogram_pos_inds = histogram_pos_inds.cuda()
        histogram_pos_repeat[histogram_pos_inds] = 0
        # Accumulating PMF to make CDF, column-wise accumulation
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        # sigma(h- * pi+)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)

        return loss

