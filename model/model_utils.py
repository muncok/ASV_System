import torch

from . import tdnnModel
from . import auxModels
from . import speechModel
from . import ResNet34

def find_model(config):
    arch = config["arch"]
    n_labels = config['n_labels']
    if arch == "SimpleCNN":
        model = auxModels.SimpleCNN(config, n_labels)
    elif arch == "gtdnn":
        model = tdnnModel.gTDNN(config, n_labels)
    elif arch == "tdnn_xvector":
        model = tdnnModel.tdnn_xvector(config, n_labels)
    elif arch == "tdnn_xvector_v1":
        model = tdnnModel.tdnn_xvector_v1(config, n_labels)
    elif arch == "CTdnnModel":
        model = tdnnModel.CTdnnModel(config, n_labels)
    elif arch == "Conv4":
        model = auxModels.Conv4(config, n_labels)
    elif arch == "ResNet34":
        model = ResNet34.ResNet34(config, 16, n_labels)
    elif arch == "ResNet34_v1":
        model = ResNet34.ResNet34_v1(config, 16, n_labels)
    elif arch == "sphere20a":
        model = auxModels.sphere20a(config, n_labels)
    elif arch == "speech_res15":
        model = speechModel.SpeechResModel("res15", n_labels)
    else:
        raise NotImplementedError

    if len(config['gpu_no']) > 1:
        print("Using gpus: {}".format(config['gpu_no']))
        model = torch.nn.DataParallel(model, device_ids=config['gpu_no'],
                output_device=config['gpu_no'][0])

    if not config["no_cuda"]:
        if len(config['gpu_no']) > 1:
            model.cuda()
        else:
            model.cuda()

    return model
