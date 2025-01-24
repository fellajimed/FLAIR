import json
from os import PathLike
import numpy as np
from sklearn.metrics import confusion_matrix


class MetricsTracker:
    def __init__(self, path: PathLike | None = None,
                 save_at_update: bool = False) -> None:
        self.values = None
        self.path = path
        self.save_at_update = save_at_update and (path is not None)

    def update(self, values_epoch: dict) -> None:
        if self.values is None:
            self.values = {k: [v] for (k, v) in values_epoch.items()}
        else:
            for (k, v) in values_epoch.items():
                self.values[k].append(v)

        if self.save_at_update:
            self.save(self.path)

    def save(self, path: PathLike) -> None:
        with open(path, 'w') as f:
            json.dump(self.values, f, indent=2)


class Metrics:
    def __init__(self, nb_classes: int = 19) -> None:
        self.nb_classes = nb_classes
        self.cm = np.zeros((nb_classes, nb_classes), dtype=int)
        self.total_loss = 0.

    def update(self, targets: np.ndarray,
               preds: np.ndarray, loss: float) -> None:
        assert targets.shape == preds.shape and len(preds.shape) == 1

        self.cm += confusion_matrix(targets, preds, normalize=None,
                                    labels=np.arange(self.nb_classes))
        self.total_loss += loss * np.prod(preds.shape)

    def summary(self) -> dict[str, float]:
        tp = np.diag(self.cm)
        fp = self.cm.sum(axis=0) - tp
        fn = self.cm.sum(axis=1) - tp
        total_pixels = self.cm.sum()

        # per-class metrics
        # NOTE: the `+ 1e-8` is to avoid nan values
        accuracy_per_class = tp / (self.cm.sum(axis=1) + 1e-8)
        precision_per_class = tp / (tp + fp + 1e-8)
        recall_per_class = tp / (tp + fn + 1e-8)
        f1_score_per_class = ((2 * precision_per_class * recall_per_class) /
                              (precision_per_class + recall_per_class + 1e-8))
        iou_per_class = tp / (tp + fp + fn + 1e-8)

        # NOTE: overall_accuracy is the weighted accuracy
        metrics = {
            "loss": (self.total_loss / total_pixels).item(),
        }

        weights = self.cm.sum(axis=1) / total_pixels
        uniform = np.ones_like(weights) / len(weights)
        per_class = {
            "_accuracy": accuracy_per_class,
            "_precision": precision_per_class,
            "_recall": recall_per_class,
            "_f1_score": f1_score_per_class,
            "_iou": iou_per_class,
        }

        for (pref, w) in (('weighted', weights), ('average', uniform)):
            for (key, val) in per_class.items():
                metrics[f"{pref}{key}"] = np.sum(val * w).item()

        return metrics
