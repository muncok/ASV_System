
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

from .data import manage_audio
from .data import dataset as dset
from .data import dataloader as dloader

def embed(config, model, audio_path):
    data = manage_audio.preprocess_from_path(config, audio_path)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    data_in = Variable(data, requires_grad=False).unsqueeze(0)
    if not config["no_cuda"]:
        data_in = data_in.cuda()
    feature = model.embed(data_in).cpu().data.numpy()
    return feature

def enroll_uttr(config, model, test_loader=None):
    if not test_loader:
        _, _, test_set = dset.SpeechDataset.read_manifest(config)
        test_loader = data.DataLoader(test_set, batch_size=config['batch_size'])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    embed_size = model.feat_size
    embeds = np.zeros([embed_size])
    for enroll_in, labels in test_loader:
        enroll_in = Variable(enroll_in, requires_grad=False)
        if not config["no_cuda"]:
            enroll_in = enroll_in.cuda()
        feature = model.embed(enroll_in)
        numeric_feature = feature.cpu().data.numpy()
        # accumulates features
        embeds += np.sum(numeric_feature, axis=0)

    # averaging the features for making signatures
    spk_models= embeds / len(test_loader.dataset.audio_labels)
    return spk_models

def enroll_frame(config, model, test_loader=None, _collate_fn=None):
    if not test_loader:
        datasets = dset.SpeechDataset.read_manifest(config)
        _,_,test_loader = dloader.get_loader(config, datasets, _collate_fn)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    embed_size = model.feat_size
    embeds = np.zeros([embed_size])
    counts = 0
    for enroll_total, labels in test_loader:
        counts += len(enroll_total)
        for i in range(0,len(enroll_total), 64):
            enroll_in = Variable(enroll_total[i:i+64], requires_grad=False)
            if not config["no_cuda"]:
                enroll_in = enroll_in.cuda()
            feature = model.embed(enroll_in)
            numeric_feature = feature.cpu().data.numpy()
            # accumulates features
            embeds += np.sum(numeric_feature, axis=0)
    # averaging the features for making signatures
    spk_models= embeds / counts
    return spk_models

def dvector(config, model, audio_path):
    data = manage_audio.preprocess_from_path(config, audio_path)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()

    uttr_data = []
    half_splice = config['splice_length'] // 2
    points = np.arange(half_splice, len(data) - half_splice)
    for point in points:
        uttr_data.append(data[point-half_splice:point+half_splice])
    uttr_data = torch.stack(uttr_data)

    feature = np.zeros((model.feat_size,))
    for i in range(0,len(uttr_data)//1, 64):
        frame_in = Variable(uttr_data[i:i+64], requires_grad=False)
        if not config["no_cuda"]:
            frame_in = frame_in.cuda()
        feature +=  np.sum(model.embed(frame_in).cpu().data.numpy(), 0)
    feature = feature / (len(uttr_data)//1)
    return feature

