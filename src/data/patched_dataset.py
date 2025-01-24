import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


DATA_AUG = transforms.Compose([
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
])


class PatchedDataset(Dataset):
    def __init__(self, dataset: Dataset, k_patches: int,
                 train: bool = False) -> None:
        self.original_dataset = dataset
        self.train = train
        self.k = k_patches
        self.coef = k_patches * k_patches
        self.new_length = len(dataset) * self.coef

    def __len__(self) -> int:
        return self.new_length

    def get_patch(self, tensor: torch.Tensor, patch_idx: int,
                  patch_height: int, patch_width: int) -> torch.Tensor:
        patches = (tensor
                   .unfold(1, patch_height, patch_height)
                   .unfold(2, patch_width, patch_width))

        patch_row = patch_idx // self.k
        patch_col = patch_idx % self.k
        return patches[:, patch_row, patch_col]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        original_idx = idx // self.coef
        patch_idx = idx % self.coef

        image, label = self.original_dataset[original_idx]

        _, height, width = image.shape

        patch_height = height // self.k
        patch_width = width // self.k

        p_image = self.get_patch(image, patch_idx, patch_height, patch_width)
        p_label = self.get_patch(label, patch_idx, patch_height, patch_width)

        if self.train:
            p_image, p_label = DATA_AUG(p_image, p_label)

        return (p_image, p_label)
