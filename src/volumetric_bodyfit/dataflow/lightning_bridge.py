import logging
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Union

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from volumetric_bodyfit.config.runtime import project_home
from nn_core.nn_types import Split


PROJECT_ROOT = Path(project_home)
pylogger = logging.getLogger(__name__)


class DatasetProfile:
    def __init__(self, schema: Mapping[str, Any]):
        self._schema = dict(schema)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "DatasetProfile":
        provider: Callable[[], Mapping[str, Any]] = getattr(dataset, "shape_spec")
        return cls(provider())

    def shape_spec(self) -> Mapping[str, Any]:
        return self._schema

    def save(self, dst_path: Path) -> None:
        pylogger.debug("Dataset profile save requested for '%s'", dst_path)

    @staticmethod
    def load(src_path: Path) -> "DatasetProfile":
        pylogger.debug("Dataset profile load requested for '%s'", src_path)
        return DatasetProfile({})


def pack_training_batch(samples: List[Any], split: Split, profile: DatasetProfile):
    return default_collate(samples)


class ShapeFieldDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        overfit: bool,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = False
        self.overfit = overfit
        self.gpus = gpus

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    @cached_property
    def metadata(self) -> DatasetProfile:
        if self.train_dataset is None:
            self.setup(stage="fit")
        return DatasetProfile.from_dataset(self.train_dataset)

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit") and self.train_dataset is None:
            self.train_dataset = hydra.utils.instantiate(self.datasets.train, mode="train")
            validation_mode = "train" if self.overfit else "val"
            self.val_dataset = hydra.utils.instantiate(self.datasets.train, mode=validation_mode)

        if stage in (None, "test") and self.test_dataset is None:
            test_mode = "train" if self.overfit else "test"
            self.test_dataset = hydra.utils.instantiate(self.datasets.train, mode=test_mode)

    def _loader(self, dataset: Dataset, split: Split, shuffle: bool, batch_size: int, workers: int) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=self.pin_memory,
            collate_fn=partial(pack_training_batch, split=split, profile=self.metadata),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(
            self.train_dataset,
            split="train",
            shuffle=not self.overfit,
            batch_size=self.batch_size.train,
            workers=self.num_workers.train,
        )

    def val_dataloader(self) -> DataLoader:
        return self._loader(
            self.val_dataset,
            split="val",
            shuffle=False,
            batch_size=self.batch_size.val,
            workers=self.num_workers.val,
        )

    def test_dataloader(self) -> DataLoader:
        return self._loader(
            self.test_dataset,
            split="test",
            shuffle=True,
            batch_size=self.batch_size.test,
            workers=self.num_workers.test,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(datasets={self.datasets}, batch_size={self.batch_size})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf_ifnet"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    hydra.utils.instantiate(cfg.nn.data, _recursive_=False)


if __name__ == "__main__":
    main()

