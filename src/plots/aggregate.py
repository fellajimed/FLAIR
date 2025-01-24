import json
from matplotlib.figure import Figure
from pathlib import Path

from .utils import fig_subplots


def plot_metrics(directory: Path) -> Figure:
    all_sets = dict()
    for json_file in directory.rglob('*.json'):
        with open(json_file, 'r') as f:
            all_sets[json_file.stem] = json.load(f)

    n_metrics = [len(metrics) for metrics in all_sets.values()]
    assert min(n_metrics) == max(n_metrics)
    n_metrics = n_metrics[0]
    keys = list(list(all_sets.values())[0].keys())

    fig, axes = fig_subplots(n_metrics, ncols=3, axsize=(6, 2.5))
    for (ax, key) in zip(axes, keys):
        for (label, metrics) in all_sets.items():
            values = metrics[key]
            if len(values) == 1:
                ax.axhline(y=values[0], linewidth=2, label=label, color='g')
            else:
                ax.plot(values, label=label)
        ax.set_title(key.replace('_', ' ').title().replace('Iou', 'IoU'))
        ax.legend()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    # argparser
    parser = argparse.ArgumentParser('aggregate')
    parser.add_argument('-d', '--directory', type=str)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    args.directory = Path(args.directory).resolve().absolute()
    if args.directory.is_file():
        args.directory = args.directory.parent

    fig = plot_metrics(args.directory)

    if args.save:
        fig.savefig(args.directory / 'aggregation_vis.pdf')
