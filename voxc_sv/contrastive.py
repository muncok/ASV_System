import torch
import torch.nn
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.

    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        #diff = x0 - x1
        #dist_sq = torch.sum(torch.pow(diff, 2), 1)
        #dist = torch.sqrt(dist_sq)

        #mdist = self.margin - dist
        #dist = torch.clamp(mdist, min=0.0)
        #loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        #loss = torch.sum(loss) / 2.0 / x0.size()[0]
        #return loss
        euclidean_distance = F.pairwise_distance(x0, x1)
        loss_contrastive = 0.5 * torch.mean((1-y) * torch.pow(euclidean_distance, 2) +
        (y) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

