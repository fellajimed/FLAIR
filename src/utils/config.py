from os import PathLike
from ruamel import yaml


def read_config(path: PathLike) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
