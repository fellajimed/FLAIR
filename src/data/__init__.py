from .flair import FLAIR
from .patched_dataset import PatchedDataset


URL = ("https://storage.gra.cloud.ovh.net/v1"
       "/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_1")

DATA_URLS = dict(
    all=(
        f"{URL}/flair_aerial_train.zip",
        f"{URL}/flair_1_aerial_test.zip",
        f"{URL}/flair_labels_train.zip",
        f"{URL}/flair_1_labels_test.zip",
    ),
    toy_dataset=(
        f"{URL}/flair_1_toy_dataset.zip",
    ),
)


__all__ = ['FLAIR', 'PatchedDataset']
