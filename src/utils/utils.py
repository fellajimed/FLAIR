import random
import torch
import numpy as np


def setup_seed(seed: int = 42) -> None:
    """ fix random seed for reproducibility """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(use_cpu: bool = False) -> torch.device:
    """ function to return the available device """
    if use_cpu:
        return torch.device('cpu')

    _device = ('cuda' if torch.cuda.is_available() else
               'mps' if torch.backends.mps.is_available() else 'cpu')

    return torch.device(_device)
