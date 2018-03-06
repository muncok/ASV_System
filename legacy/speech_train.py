# coding=utf-8
import numpy as np
import scipy.stats as st
from tqdm import tqdm, tqdm_notebook
from time import sleep

import torch
from torch.autograd import Variable
import dnn.train.model as mod
from dnn.data.dataset import protoDataset

from dnn.data.prototypical_batch_sampler import PrototypicalBatchSampler
from dnn.data.verification_batch_sampler import VerificationBatchSampler
from dnn.train.prototypical_loss import prototypical_loss as p_loss
from dnn.train.verification_loss import verification_loss as v_loss
from dnn.train.verification_loss import verification_score as eer_score
from dnn.train.verification_loss import verification_optimal_score as opt_score
from dnn.parser import get_sv_parser as get_parser

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def init_proto_loaders(opt):
    '''
    Initialize the datasets, samplers and dataloaders
    '''
    train_dataset, val_dataset = protoDataset.read_train_manifest(opt)

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
                                                num_workers=16)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_sampler=val_sampler,
                                                 num_workers=8)
    return tr_dataloader, val_dataloader

def init_sv_loaders(opt):
    '''
    Initialize the datasets, samplers and dataloaders for speaker verification
    '''
    train_dataset, val_dataset = protoDataset.read_train_manifest(opt)


    tr_sampler = VerificationBatchSampler(labels=train_dataset.audio_labels,
                                          classes_per_it=opt.classes_per_it_tr,
                                          num_support=opt.num_support_tr,
                                          num_query=opt.num_query_tr,
                                          iterations=opt.iterations)

    val_sampler = VerificationBatchSampler(labels=val_dataset.audio_labels,
                                           classes_per_it=opt.classes_per_it_val,
                                           num_support=opt.num_support_val,
                                           num_query=opt.num_query_val,
                                           iterations=opt.iterations)

    tr_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_sampler=tr_sampler,
                                                num_workers=16)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_sampler=val_sampler,
                                                 num_workers=8)
    return tr_dataloader, val_dataloader

def init_default_loaders(opt):
    '''
    Initialize the datasets, samplers and dataloaders for speaker verification
    '''
    train_dataset, val_dataset = protoDataset.read_train_manifest(opt)

    tr_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=64,
                                                num_workers=16)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=64,
                                                 num_workers=8)
    return tr_dataloader, val_dataloader

def init_speechnet(opt):
    import model as mod
    model_name = "res8-wide"
    config = mod.find_config(model_name)
    config["n_labels"] = 1211
    model_class = mod.find_model(model_name)
    model = model_class(config)
    if opt.input != None:
        model_path = opt.input
        to_state = model.state_dict()
        from_state = torch.load(model_path)
        valid_state = {k:v for k,v in from_state.items() if k in to_state.keys()}
        to_state.update(valid_state)
        model.load_state_dict(to_state)
        print("{} is loaded".format(model_path))
    model = model.cuda() if opt.cuda else model
    return model

def init_protonet(opt, fine_tune=False):
    '''
    Initialize the pre-trained resnet
    '''
    model = mod.SimpleCNN()

    if opt.input != None:
        model_path = opt.input
        to_state = model.state_dict()
        from_state = torch.load(model_path)
        valid_state = {k:v for k,v in from_state.items() if k in to_state.keys()}
        to_state.update(valid_state)
        model.load_state_dict(to_state)
        print("{} is loaded".format(model_path))

    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.convb_4.parameters():
            param.requires_grad = True

    model = model.cuda() if opt.cuda else model
    return model

def init_fullnet(opt):
    '''
    Initialize the pre-trained resnet
    '''
    import torch.nn as nn
    model = mod.SimpleCNN()
    load_state = torch.load(opt.input)
    output_shape = load_state['output.weight'].shape
    model.output = nn.Linear(output_shape[1], output_shape[0])

    model.load_state_dict(torch.load(opt.input))
    print("{} is loaded".format(opt.input))

    model = model.cuda() if opt.cuda else model
    return model

def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    learnable_parameters = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.Adam(learnable_parameters, lr=opt.learning_rate)

def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optim,
                                           step_size=opt.lr_scheduler_step,
                                           gamma=opt.lr_scheduler_gamma)

def train(opt, tr_dataloader, val_dataloader, model, optim, lr_scheduler, loss):
    '''
    Train the model with the prototypical learning algorithm
    '''
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)

        #### avg features
        # nb_splicing = opt.input_length // opt.splice_length
        # for batch in tqdm(tr_iter):
        #     x, y = batch
        #     model_outputs = []
        #     time_dim = x.size(2)
        #     split_points = range(0, time_dim-(time_dim)//nb_splicing, time_dim//nb_splicing)
        #     for point in split_points:
        #         x_in = Variable(x.narrow(2, point, time_dim//nb_splicing))
        #         if opt.cuda:
        #             x_in = x_in.cuda()
        #         model_outputs.append(model(x_in))
        #     model_output = torch.stack(model_outputs, dim=0)
        #     model_output = model_output.mean(0)
        #     y = Variable(y)
        #     if opt.cuda:
        #         y = y.cuda()
        #     l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
        #     l.backward()
        #     optim.step()
        #     train_loss.append(l.data[0])
        #     train_acc.append(acc.data[0])

        #### normal training
        for batch in tqdm(tr_iter):
            x, y = batch
            x, y = Variable(x), Variable(y)
            if opt.cuda:
                x, y = x.cuda(), y.cuda()
            model_output = model(x)
            # l, acc = p_loss(model_output, target=y, n_support=opt.num_support_tr)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
            l.backward()
            optim.step()
            train_loss.append(l.data[0])
            train_acc.append(acc.data[0])
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Train Loss: {}, Train Acc: {}'.format(avg_loss, avg_acc))
        sleep(0.05)
        val_iter = iter(val_dataloader)
        for batch in tqdm(val_iter):
            x, y = batch
            x, y = Variable(x), Variable(y)
            if opt.cuda:
                x, y = x.cuda(), y.cuda()
            model_output = model(x)
            l, acc = loss(model_output, target=y, n_support=opt.num_support_val)
            val_loss.append(l.data[0])
            val_acc.append(acc.data[0])
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc > best_acc else '(Best was {})'.format(best_acc)
        print('Val Loss: {}, Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
        if avg_acc > best_acc:
            torch.save(model.state_dict(), opt.output)
            best_acc = avg_acc

        lr_scheduler.step()

    return best_acc, train_loss, train_acc, val_loss, val_acc

# def evaluate(opt, val_dataloader, model, loss):
#     val_loss = []
#     val_acc = []
#     val_iter = iter(val_dataloader)
#     for batch in tqdm(val_iter):
#         x, y = batch
#         x, y = Variable(x), Variable(y)
#         if opt.cuda:
#             x, y = x.cuda(), y.cuda()
#         model_output = model(x)
#         l, acc = loss(model_output, target=y, n_support=opt.num_support_tr)
#         # l, acc = loss(model_output, target=y, n_classes=opt.classes_per_it_val,
#         #                 n_support=opt.num_support_tr, n_query=opt.num_query_val)
#         val_loss.append(l.data[0])
#         val_acc.append(acc.data[0])
#     avg_loss = np.mean(val_loss[-opt.iterations:])
#     avg_acc = np.mean(val_acc[-opt.iterations:])
#     print('Val Loss: {}, Val Acc: {}'.format(avg_loss, avg_acc))

def evaluate(opt, val_dataloader, model, loss, filter):
    val_loss = []
    val_acc = []
    val_iter = iter(val_dataloader)
    nb_splicing = opt.input_length // opt.splice_length
    for batch in tqdm(val_iter):
        x, y = batch
        model_outputs = []
        time_dim = x.size(2)
        split_points = range(0, time_dim-(time_dim)//nb_splicing+1, time_dim//nb_splicing)
        for point in split_points:
            x_in = Variable(x.narrow(2, point, time_dim//nb_splicing))
            if opt.cuda:
                x_in = x_in.cuda()
            embed = model.embed(x_in)
            model_outputs.append(embed)
        model_output = torch.stack(model_outputs, dim=-1)
        y = Variable(y)
        if opt.cuda:
            y = y.cuda()
        l, acc = loss(model_output, target=y, n_support=opt.num_support_tr, filter=filter)
        # l, acc = loss(model_output, target=y, n_classes=opt.classes_per_it_val,
        #                 n_support=opt.num_support_tr, n_query=opt.num_query_val)
        val_loss.append(l.data[0])
        val_acc.append(acc.data[0])
    avg_loss = np.mean(val_loss[-opt.iterations:])
    avg_acc = np.mean(val_acc[-opt.iterations:])
    print('Val Loss: {}, Val Acc: {}'.format(avg_loss, avg_acc))

def sv_score_sep(opt, val_dataloader, model):
    eer_records = []
    thresh_records = []
    val_iter = iter(val_dataloader)
    nb_splicing = opt.input_length // opt.splice_length
    n_support = opt.num_support_val
    n_classes = opt.classes_per_it_val
    n_query = opt.num_query_val
    for batch in tqdm(val_iter):
        x, y = batch

        postargs = y[:n_classes*(n_support + n_query)]
        classes = np.unique(postargs)
        def supp_idxs(c):
            return torch.LongTensor(np.where(y.numpy() == c)[0][:n_support])
        os_idxs = list(map(supp_idxs, classes))
        oq_idxs = map(lambda c: np.where(y.numpy() == c)
                                [0][n_support:], classes)
        os_x =  x[torch.stack(os_idxs).view(-1)]
        posq_x = x[np.array(list(oq_idxs)).flatten(),]
        negq_x = x[n_classes*(n_support+n_query):,]

        sup_time_dim = x.size(2)
        split_points = range(0, sup_time_dim-(sup_time_dim)//nb_splicing+1, sup_time_dim//nb_splicing)
        os_outputs = []
        for point in split_points:
            os_in = Variable(os_x.narrow(2, point, sup_time_dim//nb_splicing))
            if opt.cuda:
                os_in = os_in.cuda()
            os_outputs.append(model(os_in))
        sup_out = torch.stack(os_outputs, dim=0).mean(0)

        q_time_dim = x.size(2)//2
        split_points = range(0, q_time_dim-(q_time_dim)//nb_splicing+1, q_time_dim//nb_splicing)
        posq_outputs = []
        negq_outputs = []
        for point in split_points:
            posq_in = Variable(posq_x.narrow(2, point, q_time_dim//nb_splicing))
            negq_in = Variable(negq_x.narrow(2, point, q_time_dim//nb_splicing))
            if opt.cuda:
                posq_in = posq_in.cuda()
                negq_in = negq_in.cuda()
            posq_outputs.append(model(posq_in))
            negq_outputs.append(model(negq_in))
        posq_out = torch.stack(posq_outputs, dim=0).mean(0)
        negq_out = torch.stack(negq_outputs, dim=0).mean(0)

        y = Variable(y)
        if opt.cuda:
            y = y.cuda()
        from dnn.train.verification_loss import verification_sep_score
        eer, thresh = verification_sep_score(sup_out, posq_out, negq_out, classes)
        eer_records.append(eer)
        thresh_records.append(thresh)
    sleep(0.05)
    mean_eer = np.mean(eer_records)
    mean_thresh = np.mean(thresh_records)
    lb, ub = st.t.interval(0.95, len(eer_records)-1, loc=mean_eer, scale=st.sem(eer_records))
    lb_th, ub_th = st.t.interval(0.95, len(thresh_records)-1, loc=mean_thresh, scale=st.sem(thresh_records))
    print("eer: {:.2f}% +- {:.2f}%, thresh: {:.5f} +- {:.5f}".format(mean_eer*100, (ub-mean_eer)*100,
                                                                     mean_thresh, (ub_th-mean_thresh)))

def sv_score(opt, val_dataloader, model, filter_types, lda=None):
    val_iter = iter(val_dataloader)
    nb_splicing = opt.input_length // opt.splice_length
    eer_records = {k:[] for k in filter_types}
    thresh_records = {k:[] for k in filter_types}
    lda_eer_records = {k:[] for k in filter_types}
    lda_thresh_records = {k:[] for k in filter_types}
    model.eval()
    for batch in tqdm(val_iter):
        x, y = batch
        model_outputs = []
        time_dim = x.size(2)
        split_points = range(0, time_dim-(time_dim)//nb_splicing+1, time_dim//nb_splicing)
        for point in split_points:
            x_in = Variable(x.narrow(2, point, time_dim//nb_splicing))
            if opt.cuda:
                x_in = x_in.cuda()
            embed = model.embed(x_in)
            model_outputs.append(embed)
        model_output = torch.stack(model_outputs, dim=-1)
        if lda:
            lda_output = model_output.cpu().data.numpy()
            s, d, k = lda_output.shape
            lda_output = np.transpose(lda_output, [0,2,1]).reshape(-1, d)
            lda_output = lda.transform(lda_output).astype(np.float32)
            lda_output = lda_output.reshape(s,k,-1).transpose([0,2,1])
            if opt.cuda:
                lda_output = Variable(torch.from_numpy(lda_output).cuda())
        y = Variable(y)
        if opt.cuda:
            y = y.cuda()

        for filter_type in filter_types:
            eer, thresh = opt_score(model_output, target=y, n_classes=opt.classes_per_it_val,
                                    n_support=opt.num_support_val, n_query=opt.num_query_val,
                                    n_frames=opt.num_test_frames, filter_type=filter_type)
            eer_records[filter_type].append(eer)
            thresh_records[filter_type].append(thresh)
            sleep(0.005)
        if lda:
            for filter_type in filter_types:
                eer, thresh = opt_score(lda_output, target=y, n_classes=opt.classes_per_it_val,
                                        n_support=opt.num_support_val, n_query=opt.num_query_val,
                                        n_frames=opt.num_test_frames, filter_type=filter_type)
                lda_eer_records[filter_type].append(eer)
                lda_thresh_records[filter_type].append(thresh)
                sleep(0.005)
    for filter_type in filter_types:
        eer_record = eer_records[filter_type]
        thresh_record = thresh_records[filter_type]
        mean_eer = np.mean(eer_record)
        mean_thresh = np.mean(thresh_record)
        lb, ub = st.t.interval(0.95, len(eer_record)-1, loc=mean_eer, scale=st.sem(eer_record))
        lb_th, ub_th = st.t.interval(0.95, len(thresh_record)-1, loc=mean_thresh, scale=st.sem(thresh_record))
        print("eer[{}]: {:.2f}% +- {:.2f}%, thresh: {:.5f} +- {:.5f}".format(filter_type, mean_eer*100, (ub-mean_eer)*100,
                                                                         mean_thresh, (ub_th-mean_thresh)))
    if lda:
        for filter_type in filter_types:
            eer_record = lda_eer_records[filter_type]
            thresh_record = lda_thresh_records[filter_type]
            mean_eer = np.mean(eer_record)
            mean_thresh = np.mean(thresh_record)
            lb, ub = st.t.interval(0.95, len(eer_record)-1, loc=mean_eer, scale=st.sem(eer_record))
            lb_th, ub_th = st.t.interval(0.95, len(thresh_record)-1, loc=mean_thresh, scale=st.sem(thresh_record))
            print("lda_eer[{}]: {:.2f}% +- {:.2f}%, thresh: {:.5f} +- {:.5f}".format(filter_type, mean_eer*100, (ub-mean_eer)*100,
                                                                                 mean_thresh, (ub_th-mean_thresh)))
    return (100*mean_eer, 100*(ub-mean_eer))

def posterior_prob(opt, val_dataloader, model):
    val_iter = iter(val_dataloader)
    for batch in tqdm(val_iter):
        x, y = batch
        x_in = Variable(x)
        if opt.cuda:
            x_in = x_in.cuda()
        model_output = model(x_in)
        _, y_hat = torch.max(model_output, dim=1)

def embeds(opt, val_dataloader, model):
    val_iter = iter(val_dataloader)
    nb_splicing = opt.input_length // opt.splice_length
    model.eval()
    embeddings = []
    labels = []
    for batch in tqdm_notebook(val_iter):
        x, y = batch
        time_dim = x.size(2)
        split_points = range(0, time_dim-(time_dim)//nb_splicing+1, time_dim//nb_splicing)
        model_outputs = []
        for point in split_points:
            x_in = Variable(x.narrow(2, point, time_dim//nb_splicing))
            if opt.cuda:
                x_in = x_in.cuda()
            model_outputs.append(model.embed(x_in))
        ## snippet scale LDA
        # model_output = torch.cat(model_outputs, dim=0)
        # y = torch.cat([y]*len(model_outputs), dim=0)
        ## uttrs scale LDA, it's better now
        model_output = torch.stack(model_outputs, dim=0)
        model_output = model_output.mean(0)
        embeddings.append(model_output.cpu().data.numpy())
        labels.append(y.numpy())
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels

def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    options.train_manifest = "manifests/reddots/sv_reddots_manifest.csv"
    options.val_manifest = "manifests/reddots/sv_reddots_manifest.csv"
    # options.val_manifest = "manifests/reddots/si_reddots_train_manifest.csv"
    # options.train_manifest = "manifests/voxc/fewshot/si_voxc_train_manifest.csv"
    # options.val_manifest = "manifests/voxc/fewshot/si_voxc_val_manifest.csv"
    # options.train_manifest = "manifests/voxc/voxc_manifest.csv"

    options.n_dct_filters = 40
    options.input_length = int(16000*3.0)
    options.splice_length = int(16000*0.2)
    options.n_mels = 40
    options.timeshift_ms = 100
    options.data_folder = "/home/muncok/DL/dataset/SV_sets"
    options.window_size= 0.025
    options.window_stride= 0.010
    options.cache_size = 50000
    options.input_format = "mfcc"

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    if options.mode == "train":
        #### train ####
        tr_dataloader, val_dataloader = init_proto_loaders(options)
        print("training")
        model = init_protonet(options)
        optim = init_optim(options, model)
        lr_scheduler = init_lr_scheduler(options, optim)
        train(opt=options,
              tr_dataloader=tr_dataloader,
              val_dataloader=val_dataloader,
              model=model,
              optim=optim,
              lr_scheduler=lr_scheduler,
              loss=p_loss)
    elif options.mode == "eval":
        #### evaluate ####
        tr_dataloader, val_dataloader = init_proto_loaders(options)
        print("evaluating")
        model = init_protonet(options)
        evaluate(options, val_dataloader, model, loss=p_loss, filter=None)
        evaluate(options, val_dataloader, model, loss=p_loss, filter="diff")
    elif options.mode == "sv_score":
        #### verification scoring ####
        tr_dataloader, val_dataloader = init_sv_loaders(options)
        print("sv_scoring")
        model = init_protonet(options)
        # sv_score(options, val_dataloader, model)
        lda = pickle.load(open("models/lda/lda_voxc_0.2.pkl", "rb"))
        # sv_score(options, val_dataloader, model, ["full", "random", "diff"])
        sv_score(options, val_dataloader, model, ["full", "random", "diff", "std"], lda)
    elif options.mode == "posterior":
        print("posterior prob")
        tr_dataloader, val_dataloader = init_proto_loaders(options)
        model = init_fullnet(options)
        posterior_prob(options, val_dataloader, model)
    elif options.mode == "lda_train":
        tr_dataloader, val_dataloader = init_default_loaders(options)
        model = init_protonet(options)
        embedings, labels = embeds(options, tr_dataloader, model) # embeddings: sample x emb_size
        n_samples = embedings.shape[0]
        clf = LDA()
        random_idx = np.random.permutation(np.arange(0,n_samples))
        train_X, train_y = embedings[random_idx[:n_samples-100]], labels[random_idx[:n_samples-100]]
        test_X, test_y = embedings[random_idx[-100:]], labels[random_idx[-100:]]
        clf.fit(train_X, train_y)
        score = clf.score(test_X, test_y)
        print(score)
        pickle.dump(clf, open(options.output, "wb"))


if __name__ == '__main__':
    main()
