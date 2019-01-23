import torch

from . import tdnn
from . import resnet34

def find_model(config):
    arch = config["arch"]
    n_labels = config['n_labels']
    if arch == "tdnn_xvector":
        model = tdnn.tdnn_xvector(config, 512, n_labels)
    elif arch == "tdnn_xvector_untied":
        model = tdnn.tdnn_xvector_untied(config, 512, n_labels)
    elif arch == "ResNet34":
        model = resnet34.ResNet34(config, 16, n_labels)
    else:
        print("Not Implemented Model")
        raise NotImplementedError

    model = place_model(config, model)

    return model

def place_model(config, model):
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
