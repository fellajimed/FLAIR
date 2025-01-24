import argparse
import torch
from torch import nn
from tqdm import trange
from time import strftime
from shutil import copy2
from pathlib import Path

from . import DEFAULT_LOGS_PATH
from . import models as my_models
from .data.dataset import get_FLAIR_datasets, get_dataloaders
from .trainers.utils import train_model, test_model
from .utils.config import read_config
from .utils.utils import get_device, setup_seed
from .utils.metrics import MetricsTracker


def main() -> None:
    # argparser
    parser = argparse.ArgumentParser('main')
    parser.add_argument('config_file')
    args = parser.parse_args()

    config = read_config(args.config_file)

    # set path to store logs
    if (p := config['options'].get('logs')) is not None:
        path_logs = Path(p).resolve().absolute()
    else:
        path_logs = DEFAULT_LOGS_PATH

    date, time = strftime("%Y%m%d"), strftime("%Hh%Mmin%Ss")

    path_logs = path_logs / date / time
    path_logs.mkdir(parents=True, exist_ok=True)

    # copy the yaml in the logs directory
    print(f"The files will be saved in {str(path_logs)}")
    copy2(args.config_file, path_logs)

    # fix random seed
    seed = config['options'].get('random_seed', 42)
    setup_seed(seed)

    # get device
    use_cpu = config['options'].get('use_cpu', False)
    device = get_device(use_cpu=use_cpu)

    # datasets
    if (data_seed := config['options'].get('data_random_seed')) is not None:
        # it is possible to define a specific random seed for the data
        # this way, one could keep the same train-validation split
        # while changing the initialzation of the model
        setup_seed(data_seed)

    datasets = get_FLAIR_datasets(**config['data'])
    batch_size = config['data'].get('batches', [256, 1000, 1000])
    loaders = get_dataloaders(batch_size, *datasets, device=device)

    # fix random seed for a second time (in case it was changed for the data)
    setup_seed(seed)

    nb_classes = config['options'].get('nb_classes', 19)

    # model
    model = getattr(my_models, config['model']['cls'],
                    my_models.UNet)(**config['model']['kwargs'])
    model.to(device)
    print(model)

    # optimizer
    optimizer = getattr(torch.optim, config['optimizer']['cls'],
                        torch.optim.Adam
                        )(model.parameters(), **config['optimizer']['kwargs'])

    # loss function
    loss_fct = getattr(nn, config['loss']['cls'],
                       nn.CrossEntropyLoss)(**config['loss']['kwargs'])

    epochs = config['training'].get('epochs', 10)

    train_tracker = MetricsTracker(path_logs / 'train.json', True)
    val_tracker = MetricsTracker(path_logs / 'validation.json', True)

    for _ in trange(epochs, desc='epochs'):
        train_tracker.update(train_model(model, loaders[0], nb_classes,
                                         loss_fct, optimizer, device))

        val_tracker.update(test_model(model, loaders[1], nb_classes,
                                      loss_fct, device))

    test_tracker = MetricsTracker(path_logs / 'test.json', True)
    test_tracker.update(test_model(model, loaders[2], nb_classes,
                                   loss_fct, device))

    # save the model in the logs directory
    torch.save(model.state_dict(), path_logs / 'checkpoint.pth')


if __name__ == "__main__":
    raise SystemExit(main())
