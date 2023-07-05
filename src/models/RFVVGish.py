from typing import Any, Dict

import torch # type: ignore
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl # type: ignore

from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from conduit.data import TernarySample

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.types import Stage

class RandForestVGGish(pl.LightningModule):

    """Example of LightningModule for Random Forest embeddings.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Foward
        - Predict Step (collect embedding for train and test data)
 

    """

    def __init__(
            self,
            encoder: torch.nn.Module,
            ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['encoder'])

        self.encoder = encoder

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()


    def foward(self, x: torch.Tensor):

        # merge frames and batch to process concurrently
        # x_flat = [n x m, 1, 96, 64]
        x_flat = x.view(-1, 1, x.size(3), x.size(4))
        # perform forward pass on the batch, encode flattened
        # latent vectors z then reconstruct with decoder
        z_flat = self.encoder.forward(x_flat)
        # unflatten z,  z_flat = [n x m, 128]
        # to [n, m, 128] to separate batches from frames
        z = z_flat.view(x.size(0), -1, z_flat.size(1))
        # average z over time
        z_avg = z.mean(axis=1)
        # return results
        return z_avg


    def predict_step(self,
                 batch: Dict[Stage, TernarySample],
                 batch_idx: int) -> Dict[Stage, Dict[str, npt.NDArray]]:

        output = {}
        # for training and testing data
        for stage, sub_batch in batch.items():
            x, y, s = sub_batch
            # encode latent representation z
            z = self.forward(x.float().squeeze(1))
            # return logits and targets
            output[stage] = {"z": z.detach(),
                             "y": y.detach(),
                             "s": s }
        return output
