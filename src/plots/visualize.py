import json
from os import PathLike
from matplotlib.figure import Figure
from pathlib import Path

from .utils import fig_subplots


def plot_metrics(json_file: PathLike) -> Figure:
    with open(json_file, 'r') as f:
        metrics = json.load(f)

    fig, axes = fig_subplots(len(metrics), nrows=3, axsize=(6, 2.5))
    for (ax, (ax_title, values)) in zip(axes, metrics.items()):
        if len(values) == 1:
            ax.axhline(y=values[0], linewidth=2)
        else:
            ax.plot(values)
        ax.set_title(ax_title.replace('_', ' ').title().replace('Iou', 'IoU'))

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    # argparser
    parser = argparse.ArgumentParser('plots')
    parser.add_argument('-j', '--json-file', type=str)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    args.json_file = Path(args.json_file).resolve().absolute()

    assert args.json_file.is_file() and args.json_file.suffix == '.json'

    fig = plot_metrics(args.json_file)

    if args.save:
        fname = args.json_file.stem
        fig.savefig(args.json_file.parent / f'{fname}_vis.pdf')
