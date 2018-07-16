# import torch

from . import tdnnModel
from . import auxModels
from . import resNet34Models

# TODO: more automatic

def find_model(config, n_labels):
    arch = config["arch"]
    if arch == "SimpleCNN":
        model = auxModels.SimpleCNN(config, n_labels)
    elif arch == "TdnnModel":
        model = tdnnModel.TdnnModel(config, n_labels)
    elif arch == "CTdnnModel":
        model = tdnnModel.CTdnnModel(config, n_labels)
    elif arch == "Conv4":
        model = auxModels.Conv4(config, n_labels)
    elif arch == "ResNet34":
        model = resNet34Models.ResNet34(config, [3,4,6,3], n_labels)
    elif arch == "ResNet34_v1":
        model = resNet34Models.ResNet34_v1(config, [3,4,6,3], n_labels)
    elif arch == "ResNet34_v2":
        model = resNet34Models.ResNet34_v2(config, [3,4,6,3], n_labels)
    elif arch == "ResNet34_v3":
        model = resNet34Models.ResNet34_v3(config, [3,4,6,3], n_labels)
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

    # if len(config['gpu_no']) > 1:
        # model = torch.nn.DataParallel(model, device_ids=config['gpu_no'])

    if not config["no_cuda"]:
        model.cuda()

    return model
