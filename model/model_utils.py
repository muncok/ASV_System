import torch

from . import tdnnModel
from . import auxModels
from . import speechModel
from . import ResNet34
from . import resNet34Models
from . import wide_resnet

def find_model(config):
    arch = config["arch"]
    n_labels = config['n_labels']
    if arch == "tdnn_xvector":
        model = tdnnModel.tdnn_xvector(config, 512, n_labels)
    elif arch == "tdnn_xvector_center":
        model = tdnnModel.tdnn_xvector_center(config, 512, n_labels)
    elif arch == "tdnn_xvector_cross":
        model = tdnnModel.tdnn_xvector_cross(config, 512, n_labels)
    elif arch == "tdnn_xvector_narrow":
        model = tdnnModel.tdnn_xvector(config, 256, n_labels)
    elif arch == "tdnn_xvector_dense":
        model = tdnnModel.tdnn_xvector_dense(config, 512, n_labels)
    elif arch == "tdnn_xvector_untied":
        model = tdnnModel.tdnn_xvector_untied(config, 512, n_labels)
    elif arch == "wide_tdnn_resnet":
        # embeding dimension 512
        model = wide_resnet.Wide_Tdnn_ResNet(config['input_dim'], 10, 3, 0.3, n_labels)
    elif arch == "ResNet34":
        model = ResNet34.ResNet34(config, 16, n_labels)
    elif arch == "ResNet34_v1":
        model = ResNet34.ResNet34_v1(config, 16, n_labels, fc_dims=128)
    elif arch == "ResNet34_v3":
        model = resNet34Models.ResNet34_v3(config, [3,4,6,3], n_labels)
    elif arch == "ResNet34_v4":
        model = resNet34Models.ResNet34_v4(config, [3,4,6,3], n_labels)
    elif arch == "sphere20a":
        model = auxModels.sphere20a(config, n_labels)
    elif arch == "speech_res15":
        model = speechModel.SpeechResModel("res15", n_labels)
    elif arch == "wide_resnet":
        # embeding dimension 512
        model = wide_resnet.Wide_ResNet(10, 4, 0.3, n_labels)
    else:
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
