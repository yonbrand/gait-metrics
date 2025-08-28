#######################################################################################################################
# Model configuration
#######################################################################################################################
import copy
import torch
from pathlib import Path
from torch import nn
from models import Resnet, ElderNet

verbose = True
torch_cache_path = Path(__file__).parent / 'torch_hub_cache'
cuda = torch.cuda.is_available()


def get_config(cfg, model_type):
    if model_type is None:
        return cfg.model
    try:
        return cfg[model_type]
    except AttributeError:
        raise ValueError(f"Configuration for model type '{model_type}' not found")


def setup_model(
        net='Resnet',
        output_size=1,
        epoch_len=10,
        is_classification=False,
        is_regression=False,
        max_mu=None,
        num_layers_regressor=None,
        batch_norm=False,
        eldernet_linear_output=128,
        pretrained=False,
        trained_model_path=None,
        name_start_idx=0,
        device='cpu'):

    model = Resnet(
        output_size=output_size,
        epoch_len=epoch_len,
        is_classification=is_classification,
        is_regression=is_regression,
        max_mu=max_mu,
        num_layers_regressor=num_layers_regressor,
        batch_norm=batch_norm
    )

    if net == 'ElderNet':
        feature_extractor = model.feature_extractor
        feature_vector_size = feature_extractor[-1][0].out_channels
        model = ElderNet(feature_extractor,
                         non_linearity=True,
                         linear_model_input_size=feature_vector_size,
                         linear_model_output_size=eldernet_linear_output,
                         output_size=output_size,
                         is_classification=is_classification,
                         is_regression=is_regression,
                         max_mu=max_mu,
                         num_layers_regressor=num_layers_regressor,
                         batch_norm=batch_norm
                         )

    if pretrained or trained_model_path is not None:
        load_weights(trained_model_path, model, device, name_start_idx)

    return copy.deepcopy(model).to(device, dtype=torch.float)


def load_weights(weight_path, model, my_device="cpu", name_start_idx=0):
    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names
    model_dict = model.state_dict()

    # change the name of the keys in the pretrained model to align with our convention
    for key in pretrained_dict:
        para_names = key.split(".")
        new_key = ".".join(para_names[name_start_idx:])
        pretrained_dict_v2[new_key] = pretrained_dict_v2.pop(key)

    # 1. Check if the model has a classifier module
    has_classifier = hasattr(model, 'classifier')
    # Filter out unnecessary keys
    if has_classifier:
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict_v2.items()
            if k in model_dict
        }
    else:
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict_v2.items()
            if k in model_dict and k.split(".")[0] != "classifier"
        }
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))


def get_sslnet(harnet, tag='v1.0.0', pretrained=False, class_num=1, is_classification=False, is_regression=False):
    """
    Load and return the Self Supervised Learning (SSL) model from pytorch hub.
    :param str harnet: The pretrained network that correspond to the current input size (5/10/30 seconds)
    :param str tag: Tag on the ssl-wearables repo to check out
    :param bool pretrained: Initialise the model with UKB self-supervised pretrained weights.
    :return: pytorch SSL model
    :rtype: nn.Module
    """

    repo_name = 'ssl-wearables'
    repo = f'OxWearables/{repo_name}:{tag}'

    if not torch_cache_path.exists():
        Path.mkdir(torch_cache_path, parents=True, exist_ok=True)

    torch.hub.set_dir(str(torch_cache_path))

    # find repo cache dir that matches repo name and tag
    cache_dirs = [f for f in torch_cache_path.iterdir() if f.is_dir()]
    repo_path = next((f for f in cache_dirs if repo_name in f.name and tag in f.name), None)

    if repo_path is None:
        repo_path = repo
        source = 'github'
    else:
        repo_path = str(repo_path)
        source = 'local'
        if verbose:
            print(f'Using local {repo_path}')

    sslnet: nn.Module = torch.hub.load(repo_path, harnet, trust_repo=True, source=source, class_num=class_num,
                                       pretrained=pretrained, is_classification=is_classification,
                                       is_regression=is_regression, verbose=verbose)
    sslnet
    return sslnet
