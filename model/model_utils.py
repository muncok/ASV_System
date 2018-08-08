import torch

from . import tdnnModel
from . import auxModels
from . import resNet34Models, resNet34Models1

def find_model(config):
    arch = config["arch"]
    n_labels = config['n_labels']
    if arch == "SimpleCNN":
        model = auxModels.SimpleCNN(config, n_labels)
    elif arch == "TdnnModel":
        model = tdnnModel.TdnnModel(config, n_labels)
    elif arch == "CTdnnModel":
        model = tdnnModel.CTdnnModel(config, n_labels)
    elif arch == "Conv4":
        model = auxModels.Conv4(config, n_labels)
    elif arch == "ResNet34":
        model = resNet34Models1.ResNet34(config, 16, n_labels)
    elif arch == "ResNet34_v1":
        model = resNet34Models1.ResNet34_v1(config, 16, n_labels)
    elif arch == "ResNet34_v2":
        model = resNet34Models.ResNet34_v2(config, [3,4,6,3], n_labels)
    elif arch == "ResNet34_v3":
        model = resNet34Models.ResNet34_v3(config, [3,4,6,3], n_labels)
    elif arch == "ResNet34_v3_w":
        model = resNet34Models.ResNet34_v3_w(config, [3,4,6,3], n_labels)
    elif arch == "ResNet34_v4":
        model = resNet34Models.ResNet34_v4(config, [3,4,6,3], n_labels)
    elif arch == "ScaleResNet34":
        model = resNet34Models.ScaleResNet34(config, [3,4,6,3], n_labels)
    elif arch == "ScaleResNet34_v4":
        model = resNet34Models.ScaleResNet34_v4(config, [3,4,6,3], n_labels)
    elif arch == "sphere20a":
        model = auxModels.sphere20a(config, n_labels)
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
