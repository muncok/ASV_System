from collections import ChainMap
import argparse
import os
import random

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from .train import model as mod
from .data import dataloader as dloader

def make_abspath(rel_path):
    if not os.path.isabs(rel_path):
        rel_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), rel_path)
    return rel_path


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

def mtl_train(config, model=None):
    train_loader, dev_loader, test_loader = dloader.get_loader(config, eval=False, mtl=True)
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
        accs.append(print_eval("test", scores, labels, loss))
        accs1.append(print_eval("test", scores1, labels1, loss))
    avg_acc = np.mean(accs)
    avg_acc1 = np.mean(accs1)
    print("test accuracy: {}, {}".format(avg_acc, avg_acc1))

def train(config, loaders=None, model=None, _collate_fn=data.dataloader.default_collate):
    # train_loader, dev_loader, test_loader = dloader.get_loader(config, _collate_fn)
    if loaders is None:
        train_loader, dev_loader, test_loader = dloader.get_loader(config,_collate_fn=_collate_fn)
    else:
        train_loader, dev_loader, test_loader = loaders
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
                # scores = model(model_in)
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                accs.append(print_eval("dev", scores, labels, loss))
            avg_acc = np.mean(accs)
            print("epoch #{}, final dev accuracy: {}".format(epoch_idx,avg_acc))
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                # model.save(config["output_file"])
                torch.save(model.state_dict(), config["output_file"])
    model.eval()
    accs = []
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        accs.append(print_eval("test", scores, labels, loss))
    avg_acc = np.mean(accs)
    print("final test accuracy: {}".format(avg_acc))

def e2e_train(config, spk_model, model=None):
    train_loader, dev_loader, test_loader = dloader.get_loader(config, eval=False)
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
        accs.append(print_eval("dev", scores, labels.long(), loss,binary=True))
    avg_acc = np.mean(accs)
    print("final test accuracy: {}".format(avg_acc))

def gatedcnn_train(config, model, loaders=None):
    if loaders is None:
        train_loader, dev_loader, test_loader = dloader.get_loader(config, None, dloader._random_frames_collate_fn)
    else:
        train_loader, dev_loader, test_loader = loaders
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    learnable_params = [param for param in model.parameters() if param.requires_grad == True]
    # optimizer = torch.optim.SGD(learnable_params, lr=config["lr"][0],
                                # nesterov=config["use_nesterov"],
                                # weight_decay=config["weight_decay"], momentum=config["momentum"])
    optimizer = torch.optim.Adadelta(model.parameters())
    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    criterion = nn.NLLLoss()
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
                accs.append(print_eval("dev", scores, labels, loss))
            avg_acc = np.mean(accs)
            print("epoch #{}, final dev accuracy: {}".format(epoch_idx,avg_acc))
            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                model.save(config["output_file"])
    evaluate(config, model, test_loader)

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
