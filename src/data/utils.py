import zipfile
from tqdm import tqdm
from urllib.request import urlretrieve
from pathlib import Path
from os import PathLike


def download_and_unzip(url: str, dest: PathLike) -> None:
    zip_path, _ = urlretrieve(url)

    with zipfile.ZipFile(zip_path, "r") as f:
        for member in tqdm(f.infolist(), desc='Extracting', leave=False):
            try:
                f.extract(member, dest)
            except zipfile.error as e:
                print(e)


def download_dataset(is_toy_dataset: bool,
                     dest: str | None = None,
                     force: bool = False) -> None:
    from . import DATA_URLS
    from .. import DEFAULT_DATA_PATH

    dest = (DEFAULT_DATA_PATH if dest is None
            else Path(dest).resolve().absolute())

    key = 'toy_dataset' if is_toy_dataset else 'all'
    dest = dest / key
    dest.mkdir(parents=True, exist_ok=True)

    if force or (next(dest.rglob('*.tif')) is not None):
        for url in DATA_URLS[key]:
            download_and_unzip(url, dest)


if __name__ == "__main__":
    download_dataset(True, None)
