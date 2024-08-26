from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader

DatasetName: TypeAlias = Literal[
    "clip-benchmark/wds_imagenetv2",
    "clip-benchmark/wds_imagenet-r",
    "clip-benchmark/wds_imagenet_sketch",
    "clip-benchmark/wds_objectnet",
    "clip-benchmark/wds_imagenet-a",
]

key_dict = {
    "clip-benchmark/wds_imagenetv2": {"webp": "image", "cls": "label"},
    "clip-benchmark/wds_imagenet-r": {"jpg": "image", "cls": "label"},
    "clip-benchmark/wds_imagenet_sketch": {"jpg": "image", "cls": "label"},
    "clip-benchmark/wds_objectnet": {"png": "image", "cls": "label"},
    "clip-benchmark/wds_imagenet-a": {"jpg": "image", "cls": "label"},
}

num_classes_dict = {
    "clip-benchmark/wds_imagenetv2": 1000,
    "clip-benchmark/wds_imagenet-r": 200,
    "clip-benchmark/wds_imagenet_sketch": 1000,
    "clip-benchmark/wds_objectnet": 113,
    "clip-benchmark/wds_imagenet-a": 200,
}


@dataclass
class DomainShiftDMConfig:
    dataset_name: DatasetName
    data_dir: str
    batch_size: int
    num_workers: int
    transform: Any


class DomainShiftDataModule(L.LightningDataModule):
    def __init__(self, cfg: DomainShiftDMConfig):
        super().__init__()
        self.dataset_name = cfg.dataset_name
        self.data_dir = cfg.data_dir
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.transform = cfg.transform

    def setup(self, stage=None):
        self.dataset = load_dataset(
            self.dataset_name, cache_dir=self.data_dir, split="test"
        )

        for origin, target in key_dict[self.dataset_name].items():
            self.dataset = self.dataset.rename_column(origin, target)

        def apply_transform(batch):
            batch["image"] = [self.transform(x.convert("RGB")) for x in batch["image"]]
            return batch

        self.dataset.set_transform(apply_transform)
        self.dataset.with_format("torch")

    def test_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
