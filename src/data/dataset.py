import torch
from itertools import cycle
from os import PathLike
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from collections import Counter

from . import FLAIR, PatchedDataset
from .. import DEFAULT_DATA_PATH


def get_dataloaders(batch_size,
                    train_dataset=None,
                    validation_dataset=None,
                    test_dataset=None,
                    device=torch.device('cpu')) -> tuple[DataLoader | None]:
    '''
    function to return the train, validation and test dataloaders
    based on the batch size.
    for the train dataloader, shuffle is set to True while
    for the validation and test dataloaders is set to False
    if a dataset is None, the returned 'dataloader' is also None
    NB: batch_size is either an int or an iterable of ints
    '''
    kwargs_loader = (dict() if device == torch.device('cpu')
                     else dict(num_workers=1, pin_memory=True))

    datasets = (train_dataset, validation_dataset, test_dataset)
    batch_sizes = ([batch_size for _ in range(len(datasets))]
                   if isinstance(batch_size, int) else cycle(batch_size))

    return tuple(
        None if dataset is None
        else DataLoader(dataset, batch_size=batch_size,
                        shuffle=(i == 0), **kwargs_loader)
        for i, (dataset, batch_size) in enumerate(zip(datasets, batch_sizes)))


def get_FLAIR_datasets(*, is_toy_dataset: bool = True,
                       k_patches: int = 1,
                       ratio_train_val: float = 0.2,
                       data_path: PathLike | None = None,
                       force: bool = False,
                       **kwargs) -> tuple[PatchedDataset]:
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    else:
        data_path = Path(data_path).resolve().absolute()

    train_val_set = FLAIR(is_toy_dataset, train=True,
                          data_path=data_path, force=force)
    test_set = FLAIR(is_toy_dataset, train=False,
                     data_path=data_path, force=force)

    # split train and validation set
    train_set, val_set = random_split(
        train_val_set, [1 - ratio_train_val, ratio_train_val])

    train_set = PatchedDataset(train_set, k_patches, True)
    val_set = PatchedDataset(val_set, k_patches, False)
    test_set = PatchedDataset(test_set, k_patches, False)

    return (train_set, val_set, test_set)


def count_labels_occurrences(dataset: Dataset) -> Counter:
    cnt = Counter()
    for (_, masks) in dataset:
        values, counts = masks.unique(return_counts=True)
        cnt.update(dict(zip(values.tolist(), counts.tolist())))

    return cnt


def compute_mean_std_dataset(loader: DataLoader) -> tuple[torch.Tensor]:
    mean, mean_2 = 0., 0.
    n = 0

    for data, _ in loader:
        n += data.shape[0]
        reshaped_data = data.view(*data.shape[:2], -1)
        mean += reshaped_data.mean(dim=2).sum(dim=0)
        mean_2 += (reshaped_data**2).mean(dim=2).sum(dim=0)

    mean /= n
    mean_2 /= n

    return mean, torch.sqrt(mean_2 - mean**2)


if __name__ == "__main__":
    import argparse
    from itertools import chain

    from ..utils.config import read_config
    from ..plots.utils import fig_subplots

    # argparser
    parser = argparse.ArgumentParser('data')
    parser.add_argument('config_file')
    args = parser.parse_args()

    config = read_config(args.config_file)

    datasets = get_FLAIR_datasets(**config['data'])
    counters = []

    for (name, dataset) in zip(('train', 'validation', 'test'), datasets):
        cnt = count_labels_occurrences(dataset)
        d_cnt = dict(
            sorted([(x, f"{y/cnt.total():.2%}") for (x, y) in cnt.items()],
                   key=lambda x: x[0]))
        print(f"dataset={name} - {len(dataset)=} - total number "
              f"of pixels={cnt.total():_} - count={d_cnt}", end='\n\n')
        counters.append(cnt)

    nb_classes = 19
    counter_train_val = [counters[0].get(i, 0) + counters[1].get(i, 0)
                         for i in range(nb_classes)]
    counter_test = [counters[2].get(i, 0) for i in range(nb_classes)]

    print("Count train-val set:", counters[0] + counters[1])
    print("Count test set:", counters[2])

    fig, axes = fig_subplots(2, ncols=1, axsize=(14, 3))
    axes[0].bar(range(nb_classes), counter_train_val)
    axes[0].set_title('Train and validation sets')
    axes[0].set_xticks(list(range(nb_classes)), list(range(nb_classes)))
    axes[1].bar(range(nb_classes), counter_test)
    axes[1].set_title('Test set')
    axes[1].set_xticks(list(range(nb_classes)), list(range(nb_classes)))
    fig.tight_layout()
    fig.savefig('classes_counts.pdf')

    loaders = get_dataloaders(1000, *datasets)
    for (name, loader) in zip(['train', 'val', 'test'], loaders):
        mean, std = compute_mean_std_dataset(loader)
        print(f'{name}: mean={mean} - std={std}')
    mean, std = compute_mean_std_dataset(chain(*loaders[:-1]))
    print(f'train-val: mean={mean} - std={std}')
    mean, std = compute_mean_std_dataset(chain(*loaders))
    print(f'all: mean={mean} - std={std}')
