import torch

from . import tdnnModel
from . import auxModels
from . import resNet34Models

def find_model(config, model_type, n_labels):
    if model_type == "SimpleCNN":
        model = auxModels.SimpleCNN(config, n_labels)
    elif model_type == "TdnnModel":
        model = tdnnModel.TdnnModel(config, n_labels)
    elif model_type == "CTdnnModel":
        model = tdnnModel.CTdnnModel(config, n_labels)
    elif model_type == "Conv4":
        model = auxModels.Conv4(config, n_labels)
    elif model_type == "ResNet34":
        model = resNet34Models.ResNet34(config, [3,4,6,3], n_labels)
    elif model_type == "ResNet34_v1":
        model = resNet34Models.ResNet34_v1(config, [3,4,6,3], n_labels)
    elif model_type == "ResNet34_v2":
        model = resNet34Models.ResNet34_v2(config, [3,4,6,3], n_labels)
    elif model_type == "ScaleResNet34":
        model = resNet34Models.ScaleResNet34(config, [3,4,6,3], n_labels)
    else:
        raise NotImplementedError

    if len(config['gpu_no']) > 1:
        model = torch.nn.DataParallel(model, device_ids=config['gpu_no'])

    return model
