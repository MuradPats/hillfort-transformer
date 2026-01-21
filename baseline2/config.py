from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # dataset root
    dataset_root: Path = Path("datasets/HillfortDataSet")

    # subfolders
    rgb_dir: str = "RGB"
    dtm_dir: str = "DTM"
    label_dir: str = "Label"

    # splits (files inside dataset_root)
    train_list: str = "train.txt"
    val_list: str = "test.txt"

    # training
    num_classes: int = 2  # binary by default
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    steps_per_epoch: int | None = None  # set for smoke-test
    seed: int = 1337

    # input
    use_dtm: bool = True  # RGB+DTM (4 channels); set False for RGB-only
    dtm_scale: float = 1.0  # optionally scale DTM values
