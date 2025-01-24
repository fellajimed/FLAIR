from pathlib import Path
from os import PathLike
from tifffile import imread
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

from .utils import download_dataset


# NOTE: these values result from `python -m src.data.dataset config.yaml`
DATA_NORM = transforms.Compose([
    transforms.Normalize(
        mean=(115, 120, 110, 100, 17),
        std=(55, 48, 46, 39, 30),
    ),
])


class FLAIR(Dataset):
    def __init__(self, is_toy_dataset: bool, train: bool,
                 data_path: PathLike,
                 force: bool = False) -> None:

        key = 'toy_dataset' if is_toy_dataset else 'all'
        self.path = Path(data_path).resolve().absolute() / key

        _set = 'train' if train else 'test'
        self.aerials = sorted(self.path.rglob(f'*aerial_{_set}/**/*.tif'))
        self.labels = sorted(self.path.rglob(f'*labels_{_set}/**/*.tif'))

        if force or (len(self.aerials) * len(self.labels) == 0):
            download_dataset(is_toy_dataset, data_path, True)
            self.aerials = sorted(self.path.rglob(f'*aerial_{_set}/**/*.tif'))
            self.labels = sorted(self.path.rglob(f'*labels_{_set}/**/*.tif'))

        if len(self.aerials) != len(self.labels):
            raise ValueError(
                f"The number of images ({self.aerials}) is "
                f"different from the number of masks ({self.labels})")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        # shape: C, H, W
        _aerial = torch.moveaxis(torch.from_numpy(
            imread(self.aerials[idx])).float(), -1, 0)
        # normalize the images
        _aerial = DATA_NORM(_aerial)

        # shape: 1, H, W
        _label = torch.from_numpy(
            imread(self.labels[idx], dtype=torch.long)).unsqueeze(0)
        # label - 1 to have class indices that start from 0
        return (_aerial, _label - 1)
