from pathlib import Path

import fire
import numpy as np
from numpy.typing import ArrayLike


def move_files(files: ArrayLike, target_folder: Path):
    (target_folder / "1").mkdir()
    for file in files:
        file.replace(target_folder / "1" / file.name)


def train_val_split(imgs_root: str):
    rng = np.random.default_rng(42)

    imgs_root: Path = Path(imgs_root)
    files = np.array(list(imgs_root.iterdir()))

    mask = rng.uniform(size=len(files)) < 0.8
    train_files = files[mask]
    val_files = files[~mask]

    train_folder = imgs_root.parent / "train"
    train_folder.mkdir()
    val_folder = imgs_root.parent / "val"
    val_folder.mkdir()

    move_files(train_files, train_folder)
    move_files(val_files, val_folder)

    imgs_root.rmdir()


if __name__ == "__main__":
    fire.Fire(train_val_split)
