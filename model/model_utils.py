from . import tdnnModel
from . import auxModels

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
        model = auxModels.ResNet34(config, [3,4,6,3], n_labels)
    else:
        raise NotImplementedError

    return model
