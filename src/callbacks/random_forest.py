from typing import List, Dict, Tuple, Union, Any
from numpy import typing as npt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore
from sklearn import metrics  # type: ignore

import torch  # type: ignore
import wandb  # type: ignore
import numpy as np
import pytorch_lightning as pl  # type: ignore

from matplotlib import pyplot as plt
from matplotlib.figure import Figure  # type: ignore
from matplotlib.colors import Colormap  # type: ignore

from conduit.data import TernarySample  # type: ignore
from conduit.data.datasets.audio.ecoacoustics import SoundscapeAttr  # type: ignore

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.types import Stage

class TargetsMixin:
    def categorical_targets(self):
        return {target_name: target_attrs for target_name, target_attrs in self.targets.items()
                if target_attrs["type"] == 'categorical' }

    def continuous_targets(self):
        return  {target_name: target_attrs for target_name, target_attrs in self.targets.items()
                 if target_attrs["type"] == 'continuous' }


class RandomForest(TargetsMixin, pl.callback):
    def __init__(self,
            targets: Dict[str, Dict[str, Union[str, int]]],
            num_estimators: int = 100,
            model_name: str = 'VGGish') -> None

    self.targets = targets
    self.num_estimators = num_estimators
    self.model_name = model_name

    def on_validation_batch_end(self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            output: Dict[Stage, Dict[str, torch.Tensor]],
            batch: TernarySample,
            batch_idx: int,
            dataloader_idx: int) -> None:
        pass

    def on_validation_epoch_end(self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule) -> None:
        pass

    def on_predict_batch_end(self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            output: Dict[Stage, Dict[str, torch.Tensor]],
            batch: TernarySample,
            batch_idx: int,
            dataloader_idx: int) -> None:
        pass

    def on_predict_epoch_end(self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            outputs) -> None:
        pass

    def fit(self,
            data: Tuple[npt.NDArray, npt.NDArray]) -> Dict[int, Union[RandomForestClassifier,
                RandomForestRegressor]]
        pass

    def predict(self,
            models: Dict[int, Union[RandomForestClassifier, RandomForestRegressor]],
            data: Tuple[npt.NDArray, npt.NDArray]) -> Dict[str, Any]
    
    def preprocess(self,
            outputs: Dict[Stage, Dict[str, List]]) -> Tuple[Tuple[npt.NDArray, npt.NDArray],
                    Tuple[npt.NDArray, npt.NDArray]]:
        pass
