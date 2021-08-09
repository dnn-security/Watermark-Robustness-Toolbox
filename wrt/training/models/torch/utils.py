from typing import List

import mlconfig

import torch


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

@mlconfig.register
def load_model(model, optimizer, paths: List[str]):
    for path in paths:
        model.load_state_dict(torch.load(path)["model"])
        optimizer.load_state_dict(torch.load(path)["optimizer"])
        yield model