
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable

from . import manage_audio
from . import dataset as dset
from . import dataloader as dloader

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

# def kws_sv_system(config, signatures=None, model=None, test_loader=None):
    # if not test_loader:
        # _, _, test_set = mod.SpeechDataset.read_manifest(config)
        # test_loader = data.DataLoader(test_set, batch_size=len(test_set))
    # if not model:
        # model_name = mod.ConfigType.CNN_ONE_STRIDE1
        # kws_config = mod.find_config(model_name)
        # kws_config['n_labels'] = 4
        # kws = mod.find_model(model_name)(kws_config)
        # kws.load("model/small_onestride1.pt")

        # model_name = mod.ConfigType.RES15
        # svs_config = mod.find_config(model_name)
        # svs_config['n_labels'] = 1002
        # svs = mod.find_model(model_name)(svs_config)
        # svs.load("model/big_svs.pt")
    # if not config["no_cuda"]:
        # torch.cuda.set_device(config["gpu_no"])
        # kws.cuda()
        # svs.cuda()

    # if not signatures:
        # config['mode'] = 'enroll'
        # _, _, enroll_set = mod.SpeechDataset.read_manifest(config)
        # enroll_loader = data.DataLoader(enroll_set, batch_size=len(enroll_set))
        # signatures = enroll(config, svs, enroll_loader)
    # kws.eval()
    # svs.eval()
    # criterion = nn.CrossEntropyLoss()
    # results = []
    # total = 0
    # for kws_in, labels in test_loader:
        # # first keyword spotter
        # kws_in = Variable(kws_in, requires_grad=False)
        # if not config["no_cuda"]:
            # kws_in = kws_in.cuda()
        # kws_out = kws(kws_in)
        # scores = torch.max(kws_out, 1)[1].cpu().data

        # pass_to_svs = torch.nonzero(scores == 2)
        # kws_preds = torch.zeros(scores.shape).long()
        # kws_preds [torch.nonzero(scores == 2), ] = 1 # 2: valid keyword
        # kws_corrects = torch.sum(kws_preds == labels)
        # corrects = [i for i,on in enumerate(test_loader.dataset.audio_files) if '/on/' in on]
        # incorrects = [i for i,on in enumerate(test_loader.dataset.audio_files) if '/on/' not in on]
        # kws_TP = torch.sum(kws_preds[corrects, ] == 1) / len(corrects)
        # kws_FN = 1 - kws_TP
        # kws_FP = torch.sum(kws_preds[incorrects, ] == 1) / (len(labels) - len(incorrects))
        # kws_TN = 1 - kws_FP
        # print("KWS TP:{:.2f}, FP:{:.2f}, TN:{:.2f}, FN:{:.2f}".format(kws_TP, kws_FP, kws_TN, kws_FN))
        # print("KWS pass: {}".format(len(pass_to_svs)))

        # # speaker filtering
        # svs_in = kws_in.cpu().data[pass_to_svs,]
        # labels = labels.cpu()[pass_to_svs,].squeeze(1)
        # svs_in = Variable(svs_in, requires_grad=False)
        # if not config["no_cuda"]:
            # svs_in = svs_in.cuda()

        # # speaker verification
        # svs_out = svs(svs_in, feature=True)
        # test_sigs = svs_out.cpu().data
        # svs_preds = torch.zeros(labels.shape).long()
        # threshold = 0.66
        # for i, sig in enumerate(test_sigs):
            # test_sig = sig.unsqueeze(0)
            # max_similarity = torch.max(F.cosine_similarity(test_sig, signatures))
            # if max_similarity > threshold:
                # svs_preds[i] = 1
        # svs_corrects = torch.sum(svs_preds == labels)
        # print("SVS Acc: {}/{}".format(svs_corrects, labels.size(0)))

        # # labels = Variable(labels, requires_grad=False)
        # # loss = criterion(svs_out, labels)
        # # results.append(print_eval("test", svs_out, labels, loss) * kws_in.size(0))
        # # total += kws_in.size(0)
    # # print("final test accuracy: {}".format(sum(results) / total))
    # # second keyword spotter
