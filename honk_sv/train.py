from collections import ChainMap
import argparse
import os
import random
import sys

import librosa
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from . import model as mod
from .manage_audio import preprocess_audio

def make_abspath(rel_path):
    if not os.path.isabs(rel_path):
        rel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel_path)
    return rel_path

class ConfigBuilder(object):
    def __init__(self, *default_configs):
        self.default_config = ChainMap(*default_configs)

    def build_argparse(self):
        parser = argparse.ArgumentParser()
        for key, value in self.default_config.items():
            key = "--{}".format(key)
            # default_config's values are inserted through default argument
            if isinstance(value, tuple):
                parser.add_argument(key, default=list(value), nargs=len(value), type=type(value[0]))
            elif isinstance(value, list):
                parser.add_argument(key, default=value, nargs="+", type=type(value[0]))
            elif isinstance(value, bool) and not value:
                parser.add_argument(key, action="store_true")
            else:
                parser.add_argument(key, default=value, type=type(value))
        return parser

    def config_from_argparse(self, parser=None):
        if not parser:
            parser = self.build_argparse()
        args = vars(parser.parse_known_args()[0])
        return ChainMap(args, self.default_config)

def print_eval(name, scores, labels, loss, end="\n", verbose=False, binary=False):
    batch_size = labels.size(0)
    if not binary:
        accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).sum() / batch_size
    else:
        preds = (scores.data > 0.5)
        targets = (labels.data == 1)
        accuracy = (preds == targets).sum() / batch_size
    loss = loss.cpu().data.numpy()[0]
    if verbose:
        print("{} accuracy: {:>3}, loss: {:<7}".format(name, accuracy, loss), end=end)
    return accuracy

def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def embed(config, model, audio):
    data = librosa.core.load(audio, sr=16000)[0]
    in_len = config['input_length']
    n_mels = config['n_mels']
    filters = librosa.filters.dct(config["n_dct_filters"], config["n_mels"])
    if len(data) > in_len:
        start_frame = np.random.randint(0, len(data) - in_len)
        data = data[start_frame:start_frame+in_len]
    else:
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
    data = torch.from_numpy(preprocess_audio(data, n_mels, filters))

    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    data_in = Variable(data, requires_grad=False).unsqueeze(0)
    if not config["no_cuda"]:
        data_in = data_in.cuda()
    feature = model(data_in, feature=True).cpu().data.numpy()
    return feature


def enroll(config, model, test_loader=None):
    if not test_loader:
        _,_, test_loader = get_loader(config, _frames_collate_fn)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    embed_size = model.output.in_features
    embeds = np.zeros([embed_size])
    for enroll_in, labels in test_loader:
        enroll_in = Variable(enroll_in, requires_grad=False)
        if not config["no_cuda"]:
            enroll_in = enroll_in.cuda()
        feature = model(enroll_in, feature=True)
        numeric_feature = feature.cpu().data.numpy()
        # accumulates features
        embeds += np.sum(numeric_feature, axis=0)

    # averaging the features for making signatures
    spk_models= embeds / len(test_loader.dataset.audio_labels)
    return spk_models

def enroll_frames(config, model, test_loader=None):
    if not test_loader:
        _,_, test_loader = get_loader(config, _all_frames_collate_fn)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    embed_size = model.output.in_features
    embeds = np.zeros([embed_size])
    counts = 0
    for enroll_total, labels in test_loader:
        counts += len(enroll_total)//4
        for i in range(0,len(enroll_total)//4, 64):
            enroll_in = Variable(enroll_total[i:i+64], requires_grad=False)
            if not config["no_cuda"]:
                enroll_in = enroll_in.cuda()
            feature = model(enroll_in, feature=True)
            numeric_feature = feature.cpu().data.numpy()
            # accumulates features
            embeds += np.sum(numeric_feature, axis=0)

    # averaging the features for making signatures
    print(counts)
    spk_models= embeds / counts

    return spk_models
def kws_sv_system(config, signatures=None, model=None, test_loader=None):
    if not test_loader:
        _, _, test_set = mod.SpeechDataset.read_manifest(config)
        test_loader = data.DataLoader(test_set, batch_size=len(test_set))
    if not model:
        model_name = mod.ConfigType.CNN_ONE_STRIDE1
        kws_config = mod.find_config(model_name)
        kws_config['n_labels'] = 4
        kws = mod.find_model(model_name)(kws_config)
        kws.load("model/small_onestride1.pt")

        model_name = mod.ConfigType.RES15
        svs_config = mod.find_config(model_name)
        svs_config['n_labels'] = 1002
        svs = mod.find_model(model_name)(svs_config)
        svs.load("model/big_svs.pt")
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        kws.cuda()
        svs.cuda()

    if not signatures:
        config['mode'] = 'enroll'
        _, _, enroll_set = mod.SpeechDataset.read_manifest(config)
        enroll_loader = data.DataLoader(enroll_set, batch_size=len(enroll_set))
        signatures = enroll(config, svs, enroll_loader)
    kws.eval()
    svs.eval()
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    for kws_in, labels in test_loader:
        # first keyword spotter
        kws_in = Variable(kws_in, requires_grad=False)
        if not config["no_cuda"]:
            kws_in = kws_in.cuda()
        kws_out = kws(kws_in)
        scores = torch.max(kws_out, 1)[1].cpu().data

        pass_to_svs = torch.nonzero(scores == 2)
        kws_preds = torch.zeros(scores.shape).long()
        kws_preds [torch.nonzero(scores == 2), ] = 1 # 2: valid keyword
        kws_corrects = torch.sum(kws_preds == labels)
        corrects = [i for i,on in enumerate(test_loader.dataset.audio_files) if '/on/' in on]
        incorrects = [i for i,on in enumerate(test_loader.dataset.audio_files) if '/on/' not in on]
        kws_TP = torch.sum(kws_preds[corrects, ] == 1) / len(corrects)
        kws_FN = 1 - kws_TP
        kws_FP = torch.sum(kws_preds[incorrects, ] == 1) / (len(labels) - len(incorrects))
        kws_TN = 1 - kws_FP
        print("KWS TP:{:.2f}, FP:{:.2f}, TN:{:.2f}, FN:{:.2f}".format(kws_TP, kws_FP, kws_TN, kws_FN))
        print("KWS pass: {}".format(len(pass_to_svs)))

        # speaker filtering
        svs_in = kws_in.cpu().data[pass_to_svs,]
        labels = labels.cpu()[pass_to_svs,].squeeze(1)
        svs_in = Variable(svs_in, requires_grad=False)
        if not config["no_cuda"]:
            svs_in = svs_in.cuda()

        # speaker verification
        svs_out = svs(svs_in, feature=True)
        test_sigs = svs_out.cpu().data
        svs_preds = torch.zeros(labels.shape).long()
        threshold = 0.66
        for i, sig in enumerate(test_sigs):
            test_sig = sig.unsqueeze(0)
            max_similarity = torch.max(F.cosine_similarity(test_sig, signatures))
            if max_similarity > threshold:
                svs_preds[i] = 1
        svs_corrects = torch.sum(svs_preds == labels)
        print("SVS Acc: {}/{}".format(svs_corrects, labels.size(0)))

        # labels = Variable(labels, requires_grad=False)
        # loss = criterion(svs_out, labels)
        # results.append(print_eval("test", svs_out, labels, loss) * kws_in.size(0))
        # total += kws_in.size(0)
    # print("final test accuracy: {}".format(sum(results) / total))
    # second keyword spotter

def evaluate(config, model=None, test_loader=None):
    if not test_loader:
        _, _, test_set = mod.SpeechDataset.read_manifest(config)
        test_loader = data.DataLoader(test_set, batch_size=config['batch_size'])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
        print("{} is loaded:".format(config["input_file"]))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        results.append(print_eval("test", scores, labels, loss,verbose=False) * model_in.size(0))
        total += model_in.size(0)
    print("final test accuracy: {}".format(sum(results) / total))

def get_loader(config, _collate_fn=data.dataloader.default_collate):
    num_workers_ = config["num_workers"]
    batch_size = config['batch_size']
    train_set, dev_set, test_set = mod.SpeechDataset.read_manifest(config)
    collate_fn_ = _collate_fn

    train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                   shuffle=True, drop_last=True, num_workers=num_workers_,
                                   collate_fn=collate_fn_)
    dev_loader = data.DataLoader(dev_set, batch_size=min(len(dev_set), batch_size), shuffle=True,
                                 num_workers=num_workers_, collate_fn=collate_fn_)
    test_loader = data.DataLoader(test_set, batch_size=min(len(test_set), batch_size), shuffle=True,
                                  num_workers=num_workers_, collate_fn=collate_fn_)
    return train_loader, dev_loader, test_loader

def _frames_collate_fn(batch):
    splice_length = 20
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

        ## all frames
        # points = np.arange(half_splice, tensor.size(0) - half_splice)
        # for point in points:
        #     tensors.append(tensor[point-half_splice:point+half_splice+1])
        #     targets.append(target)
    # from sklearn.utils import shuffle
    # tensors, targets = shuffle(tensors, targets)
    inputs = torch.stack(tensors)
    targets = torch.LongTensor(targets)
    return inputs, targets

def _all_frames_collate_fn(batch):
    splice_length = 20
    half_splice = splice_length //2
    minibatch_size = len(batch)
    tensors = []
    targets = []
    # inputs = torch.zeros(minibatch_size, splice_length, 40)
    for x in range(minibatch_size):
        sample = batch[x]
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

def mtl_train(config, model=None):
    train_loader, dev_loader, test_loader = get_loader(config, eval=False, mtl=True)
    if not model:
        model = config["model_class"](config)
        if config["input_file"]:
            model.load(config["input_file"])
            print("{} is loaded:".format(config["input_file"]))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    learnable_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][0], nesterov=config["use_nesterov"],
                                weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()
    max_acc = 0
    step_no = 0
    alpha = config['alpha']

    for epoch_idx in range(config["n_epochs"]):
        for batch_idx, (model_in, labels, labels1) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
                labels1 = labels1.cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in, task=0); scores1 = model(model_in, task=1)
            labels = Variable(labels, requires_grad=False); labels1 = Variable(labels1, requires_grad=False)
            loss = alpha * criterion(scores, labels) + (1.0 - alpha)*criterion(scores1, labels1)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                                            nesterov=config["use_nesterov"], momentum=config["momentum"],
                                            weight_decay=config["weight_decay"])
            print_step = config["print_step"]
            if step_no % print_step == print_step -1:
                print_eval("[spk] train step #{}".format(step_no), scores, labels, loss, verbose=True)
                print_eval("[sent] train step #{}".format(step_no), scores1, labels1, loss, verbose=True)

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            accs1 = []
            for model_in, labels, labels1 in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                    labels1 = labels1.cuda()
                scores = model(model_in, task=0); scores1 = model(model_in, task=1)
                labels = Variable(labels, requires_grad=False); labels1 = Variable(labels1, requires_grad=False)
                loss = criterion(scores, labels) + criterion(scores1, labels1)
                loss_numeric = loss.cpu().data.numpy()[0]
                accs.append(print_eval("dev", scores, labels, loss))
                accs1.append(print_eval("dev", scores1, labels1, loss))
            avg_acc = np.mean(accs)
            avg_acc1 = np.mean(accs1)
            print("epoch #{}, final dev accuracy: {}, {}".format(epoch_idx,avg_acc, avg_acc1))
            if min(avg_acc, avg_acc1) > max_acc:
                print("saving best model...")
                max_acc = max(avg_acc,avg_acc1)
                model.save(config["output_file"])

    accs = []
    accs1 = []
    for model_in, labels, labels1 in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
            labels1 = labels1.cuda()
        scores = model(model_in, task=0); scores1 = model(model_in, task=1)
        labels = Variable(labels, requires_grad=False); labels1 = Variable(labels1, requires_grad=False)
        loss = criterion(scores, labels) + criterion(scores1, labels1)
        loss_numeric = loss.cpu().data.numpy()[0]
        accs.append(print_eval("test", scores, labels, loss))
        accs1.append(print_eval("test", scores1, labels1, loss))
    avg_acc = np.mean(accs)
    avg_acc1 = np.mean(accs1)
    print("test accuracy: {}, {}".format(avg_acc, avg_acc1))

def train(config, model=None):
    train_loader, dev_loader, test_loader = get_loader(config, _frames_collate_fn)
    if not model:
        model = config["model_class"](config)
        if config["input_file"]:
            model.load(config["input_file"])
            print("{} is loaded:".format(config["input_file"]))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    learnable_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][0], nesterov=config["use_nesterov"],
                                weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()
    max_acc = 0
    step_no = 0

    for epoch_idx in range(config["n_epochs"]):
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if config['model'].startswith('lstm'):
                model_in.transpose_(0,1)
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"], momentum=config["momentum"],
                                            weight_decay=config["weight_decay"])
            print_step = config["print_step"]
            if step_no % print_step == print_step -1:
                print_eval("train step #{}".format(step_no), scores, labels, loss, verbose=True)

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                loss_numeric = loss.cpu().data.numpy()[0]
                accs.append(print_eval("dev", scores, labels, loss))
            avg_acc = np.mean(accs)
            print("epoch #{}, final dev accuracy: {}".format(epoch_idx,avg_acc))
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                model.save(config["output_file"])
    evaluate(config, model, test_loader)

def frame_train(config, model=None):
    train_loader, dev_loader, test_loader = get_loader(config,_collate_fn=_frames_collate_fn)
    if not model:
        model = config["model_class"](config)
        if config["input_file"]:
            model.load(config["input_file"])
            print("{} is loaded:".format(config["input_file"]))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    learnable_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][0], nesterov=config["use_nesterov"],
                                weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.CrossEntropyLoss()
    max_acc = 0
    step_no = 0
    batch_size = config["batch_size"]

    for epoch_idx in range(config["n_epochs"]):
        for super_batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if config['model'].startswith('lstm'):
                model_in.transpose_(0,1)
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            for batch_idx in range(0, len(labels), batch_size):
                batch_in = Variable(model_in[batch_idx:batch_idx+batch_size], requires_grad=False)
                scores = model(batch_in)
                batch_labels = Variable(labels[batch_idx:batch_idx+batch_size], requires_grad=False)
                loss = criterion(scores, batch_labels)
                loss.backward()
                optimizer.step()
                step_no += 1
                if step_no > schedule_steps[sched_idx]:
                    sched_idx += 1
                    print("changing learning rate to {}".format(config["lr"][sched_idx]))
                    optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][sched_idx],
                                                nesterov=config["use_nesterov"], momentum=config["momentum"],
                                                weight_decay=config["weight_decay"])
        print_step = config["print_step"]
        if step_no % print_step == print_step -1:
            print_eval("train step #{}".format(step_no), scores, batch_labels, loss, verbose=True)

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                loss_numeric = loss.cpu().data.numpy()[0]
                accs.append(print_eval("dev", scores, labels, loss))
            avg_acc = np.mean(accs)
            print("epoch #{}, final dev accuracy: {}".format(epoch_idx,avg_acc))
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                model.save(config["output_file"])
    evaluate(config, model, test_loader)
def sv_train(config, spk_model, model=None):
    train_loader, dev_loader, test_loader = get_loader(config, eval=False)
    if not model:
        model = config["model_class"](config)
        if config["input_file"]:
            model.load(config["input_file"])
            print("{} is loaded:".format(config["input_file"]))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    learnable_params = [param for param in model.parameters() if param.requires_grad == True]
    optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][0], nesterov=config["use_nesterov"],
                                weight_decay=config["weight_decay"], momentum=config["momentum"])
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.BCEWithLogitsLoss()
    max_acc = 0
    step_no = 0

    #speaker model
    spk_model_in = torch.from_numpy(spk_model).float().unsqueeze(0)
    if  not config["no_cuda"]:
        spk_model_in = spk_model_in.cuda()
    spk_model_in = Variable(spk_model_in, requires_grad=False)

    for epoch_idx in range(config["n_epochs"]):
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if config['model'].startswith('lstm'):
                model_in.transpose_(0,1)
            if not config["no_cuda"]:
                model_in = model_in.cuda()
                labels = labels.unsqueeze(1).float().cuda()
            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in, spk_model_in)
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][sched_idx],
                                            nesterov=config["use_nesterov"], momentum=config["momentum"],
                                            weight_decay=config["weight_decay"])
            print_step = config["print_step"]
            if step_no % print_step == print_step -1:
                print_eval("train step #{}".format(step_no), scores, labels.long(), loss, verbose=True, binary=True)

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.unsqueeze(1).float().cuda()
                scores = model(model_in,spk_model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                loss_numeric = loss.cpu().data.numpy()[0]
                accs.append(print_eval("dev", scores, labels.long(), loss,binary=True))
            avg_acc = np.mean(accs)
            print("epoch #{}, final dev accuracy: {}".format(epoch_idx,avg_acc))
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                model.save(config["output_file"])
        model.eval()
    accs = []
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.unsqueeze(1).float().cuda()
        scores = model(model_in,spk_model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        loss_numeric = loss.cpu().data.numpy()[0]
        accs.append(print_eval("dev", scores, labels.long(), loss,binary=True))
    avg_acc = np.mean(accs)
    print("final test accuracy: {}".format(avg_acc))


def print_config(config):
    train_config = ['batch_size', 'lr', 'schedule', 'weight_decay', 'use_nesterov',
                    'num_workers', 'no_cuda', 'use_dilation']
    model_config = ['model', 'n_feature_maps', 'n_layers', 'input_file', 'output_file']
    input_config = ['dataset', 'n_labels','train_manifest', 'val_manifest', 'test_manifest',
                    'input_length', 'n_mels', 'timeshift_ms']
    mode_config = ['mode', 'system']

    print("##### mode #####")
    for key in mode_config:
        if key in config:
            item = config[key]
            print("\t{}: {}".format(key, item))

    print("##### model #####")
    for key in model_config:
        if key in config:
            item = config[key]
            print("\t{}: {}".format(key, item))

    print("##### input #####")
    for key in input_config:
        if key in config:
            item = config[key]
            print("\t{}: {}".format(key, item))

    if config['mode'] == 'train':
        print("##### train #####")
        for key in train_config:
            if key in config:
                item = config[key]
                print("\t{}: {}".format(key, item))


def main():
    output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "model", "model.pt")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="cnn-trad-pool2", type=str)
    parser.add_argument("--dataset", choices=["voxc", "command"], default="voxc", type=str)
    config, _ = parser.parse_known_args()

    global_config = dict(model=config.model, dataset=config.dataset, no_cuda=False, n_epochs=500, lr=[0.001],
                         schedule=[np.inf], batch_size=64, dev_every=1, seed=0,
                         use_nesterov=False, input_file="", output_file=output_file, gpu_no=0, cache_size=32768,
                         momentum=0.9, weight_decay=0.00001, num_workers = 16, print_step=1)
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        mod.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)
    # print_config(config)

if __name__ == "__main__":
    main()
