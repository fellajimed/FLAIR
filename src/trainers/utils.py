import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


from ..utils.metrics import Metrics


def train_model(model: nn.Module, loader: DataLoader, nb_classes: int,
                loss_fct: nn.Module, optimizer: Optimizer, device=None):
    if device is None:
        device = next(model.parameters()).device

    metrics = Metrics(nb_classes)

    model.train()
    for (images, masks) in tqdm(loader, desc='dataloader train',
                                total=len(loader), leave=False):
        images, masks = images.to(device), masks.squeeze().to(device)

        outputs = model(images)
        loss = loss_fct(outputs, masks.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            metrics.update(masks.cpu().view(-1).numpy(),
                           outputs.argmax(dim=1).cpu().view(-1).numpy(),
                           loss.item())

    return metrics.summary()


def test_model(model: nn.Module, loader: DataLoader, nb_classes: int,
               loss_fct: nn.Module | None = None, device=None):
    if device is None:
        device = next(model.parameters()).device

    if loss_fct is None:
        loss_fct = nn.CrossEntropyLoss()

    metrics = Metrics(nb_classes)

    with torch.no_grad():
        model.eval()
        for (images, masks) in tqdm(loader, desc='dataloader eval',
                                    total=len(loader), leave=False):
            images, masks = images.to(device), masks.squeeze().to(device)

            outputs = model(images)

            metrics.update(masks.cpu().view(-1).numpy(),
                           outputs.argmax(dim=1).cpu().view(-1).numpy(),
                           loss_fct(outputs, masks.long()).item())

    return metrics.summary()
