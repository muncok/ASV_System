# coding=utf-8
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc
from scipy import interp

def compute_eer(score_vector, label_vector):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    eer = dict()
    thres = dict()
    n_classes = score_vector.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], thres[i] = roc_curve(label_vector[:, i], score_vector[:, i], pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], thres["micro"] = roc_curve(label_vector.ravel(), score_vector.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    eer["micro"] = fpr["micro"][np.nanargmin(np.abs(fpr["micro"] - (1 - tpr["micro"])))]

    for i in range(n_classes):
        eer[i] = fpr[i][np.nanargmin(np.abs(fpr[i] - (1 - tpr[i])))]

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    eer["macro"] = fpr["macro"][np.nanargmin(np.abs(fpr["macro"] - (1 - tpr["macro"])))]
    thres["micro"] = thres["micro"][np.nanargmin(np.abs(fpr["macro"] - (1 - tpr["macro"])))]
    return eer["micro"], thres["micro"]


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

def cosine_similarity(x, y):
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

    return F.cosine_similarity(x, y, dim=2)

def cosine_similarity_filter(x, y, n_frames, ispos, filter_type):
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

    # k = n_frames
    # x = x[:,:,:,:k]
    # y = y[:,:,:,:k]  # doesn't have meaning
    n_topk = int(k/2) + 1
    if filter_type == "topk":
        largest_ = True if ispos else False
        idxs = torch.topk(F.cosine_similarity(x, y, dim=2), n_topk, dim=2, largest=largest_)[1]
        return F.cosine_similarity(x.gather(3, idxs.unsqueeze(2).expand(n,m,d,-1)).mean(-1), y.mean(-1), dim=2)
    elif filter_type == "random":
        random_idx = torch.randperm(k)[:n_topk].cuda() if x.is_cuda else torch.randperm(k)[:n_topk]
        return F.cosine_similarity(x[:,:,:,random_idx].mean(-1), y.mean(-1), dim=2)
    elif filter_type == "diff":
        score = F.cosine_similarity(x, y, dim=2)  # (n,m,k)
        score = score.transpose_(1,2)  # (n,k,m)
        score_top2, top2_idxs = torch.topk(score, 2, dim=2, largest=True)  # (n,k,2)
        score_top2_diff =  score_top2[:,:,0] - score_top2[:,:,1]  # (n,k)
        _, topk_diff_idxs = torch.topk(score_top2_diff, n_topk, dim=1) # (n, topk)
        topk_diff_idxs = topk_diff_idxs.unsqueeze(1).unsqueeze(2).expand(n,m,d,-1)
        return F.cosine_similarity(x.gather(3, topk_diff_idxs).mean(-1), y.mean(-1), dim=2)
    elif filter_type == "std":
        score = F.cosine_similarity(x, y, dim=2)  # (n,m,k)
        score = score.transpose_(1,2)  # (n,k,m)
        score_std = score.std(2)  # over 3rd dimension
        topk_stds, topk_stds_idxs = torch.topk(score_std, n_topk, dim=1)
        topk_stds_idxs = topk_stds_idxs.unsqueeze(1).unsqueeze(2).expand(n,m,d,-1)
        return F.cosine_similarity(x.gather(3, topk_stds_idxs).mean(-1), y.mean(-1), dim=2)
    elif filter_type == "full":
        # no filtering
        return F.cosine_similarity(x.mean(-1), y.mean(-1), dim=2)
    else:
        raise NotImplementedError

def verification_score(input, target, n_classes, n_support, n_query):
    cputargs = target.cpu() if target.is_cuda else target
    cputargs = cputargs.data
    cpuinput = input.cpu() if target.is_cuda else input

    def supp_idxs(c):
        return torch.LongTensor(np.where(cputargs.numpy() == c)[0][:n_support])

    postargs = cputargs[:n_classes*(n_support + n_query)]
    classes = np.unique(postargs)
    # n_query = len(np.where(cputargs.numpy() == classes[0])[0]) - n_support
    os_idxs = list(map(supp_idxs, classes))
    prototypes = [cpuinput[i].mean(0).data.numpy().tolist() for i in os_idxs]
    prototypes = Variable(torch.FloatTensor(prototypes))

    oq_idxs = map(lambda c: np.where(cputargs.numpy() == c)
                            [0][n_support:], classes)
    oq = input[np.array(list(oq_idxs)).flatten(),]
    negq = input[n_classes*(n_support+n_query):,]

    prototypes = prototypes.cuda() if target.is_cuda else prototypes

    ### cosine similarity
    pos_dists = cosine_similarity(oq, prototypes)
    pos_dists = pos_dists.cpu().data if target.is_cuda else pos_dists.cpu()
    neg_dists = cosine_similarity(negq, prototypes)
    neg_dists = neg_dists.cpu().data if target.is_cuda else neg_dists.cpu()
    num_class = len(classes)
    pos_label = np.eye(num_class)[np.array([[i] * n_query
                                             for i in range(num_class)]).reshape(-1)]
    neg_label = np.zeros((len(negq), num_class))

    dists = torch.cat((pos_dists, neg_dists), 0)
    labels = np.concatenate((pos_label, neg_label), axis=0)
    eer = compute_eer(dists.numpy(), labels)

    return eer

def verification_optimal_score(inputs, target, n_classes, n_support, n_query, n_frames, filter_type):
    input = inputs.mean(2)
    cputargs = target.cpu() if target.is_cuda else target
    cputargs = cputargs.data
    cpuinput = input.cpu() if target.is_cuda else input

    def supp_idxs(c):
        return torch.LongTensor(np.where(cputargs.numpy() == c)[0][:n_support])

    postargs = cputargs[:n_classes*(n_support + n_query)]
    classes = np.unique(postargs)
    os_idxs = list(map(supp_idxs, classes))
    prototypes = [cpuinput[i].mean(0).data.numpy().tolist() for i in os_idxs]
    prototypes = Variable(torch.FloatTensor(prototypes))

    oq_idxs = map(lambda c: np.where(cputargs.numpy() == c)
                            [0][n_support:], classes)
    oq = inputs[np.array(list(oq_idxs)).flatten(),]
    negq = inputs[n_classes*(n_support+n_query):,]

    prototypes = prototypes.cuda() if target.is_cuda else prototypes

    ### cosine similarity
    pos_dists = cosine_similarity_filter(oq, prototypes, n_frames, ispos=True, filter_type=filter_type)
    pos_dists = pos_dists.cpu().data if target.is_cuda else pos_dists.cpu()
    neg_dists = cosine_similarity_filter(negq, prototypes, n_frames, ispos=False, filter_type=filter_type)
    neg_dists = neg_dists.cpu().data if target.is_cuda else neg_dists.cpu()
    num_class = len(classes)
    pos_label = np.eye(num_class)[np.array([[i] * n_query
                                            for i in range(num_class)]).reshape(-1)]
    neg_label = np.zeros((len(negq), num_class))

    dists = torch.cat((pos_dists, neg_dists), 0)
    labels = np.concatenate((pos_label, neg_label), axis=0)
    eer = compute_eer(dists.numpy(), labels)

    return eer

def verification_sep_score(sup, posq, negq, classes):

    prototypes = sup.view(len(classes), -1, sup.size(-1)).mean(1)

    ### cosine similarity
    pos_dists = cosine_similarity(posq, prototypes)
    pos_dists = pos_dists.cpu().data if pos_dists.is_cuda else pos_dists.cpu()
    neg_dists = cosine_similarity(negq, prototypes)
    neg_dists = neg_dists.cpu().data if neg_dists.is_cuda else neg_dists.cpu()
    num_class = len(classes)
    n_query = int(posq.size(0) / num_class)
    pos_label = np.eye(num_class)[np.array([[i] * n_query
                                            for i in range(num_class)]).reshape(-1)]
    neg_label = np.zeros((len(negq), num_class))

    dists = torch.cat((pos_dists, neg_dists), 0)
    labels = np.concatenate((pos_label, neg_label), axis=0)
    eer = compute_eer(dists.numpy(), labels)

    return eer

def verification_loss(input, target, n_classes, n_support, n_query, filter_type):
    cputargs = target.cpu() if target.is_cuda else target
    cputargs = cputargs.data
    cpuinput = input.cpu() if target.is_cuda else input

    def supp_idxs(c):
        return torch.LongTensor(np.where(cputargs.numpy() == c)[0][:n_support])

    postargs = cputargs[:n_classes*(n_support + n_query)]
    classes = np.unique(postargs)
    n_query = len(np.where(cputargs.numpy() == classes[0])[0]) - n_support
    os_idxs = list(map(supp_idxs, classes))
    prototypes = [cpuinput[i].mean(0).data.numpy().tolist() for i in os_idxs]
    prototypes = Variable(torch.FloatTensor(prototypes))

    posq_idxs = map(lambda c: np.where(cputargs.numpy() == c)
                            [0][n_support:], classes)
    posq = input[np.array(list(posq_idxs)).flatten(),]
    negq = input[n_classes*(n_support+n_query):,]

    prototypes = prototypes.cuda() if target.is_cuda else prototypes

    ### cosine similarity
    pdists = cosine_similarity(posq, prototypes)
    ndists = cosine_similarity(negq, prototypes)

    log_p_y = F.log_softmax(pdists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    target_inds = target_inds.long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda() if target.is_cuda else target_inds

    n_log_p_y = F.log_softmax(ndists, dim=1)
    n_loss_val = n_log_p_y.max(1)[0].mean()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean() + n_loss_val
    _, y_hat = log_p_y.max(2)

    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
    return loss_val, acc_val

