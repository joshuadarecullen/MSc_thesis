from typing import List, Tuple, Any, Union, Any, Dict, Optional
from conduit.data.datasets.audio.ecoacoustics import Ecoacoustics, SoundscapeAttr
from conduit.data.datamodules.audio.ecoacoustics import EcoacousticsDataModule
from conduit.data.datasets.utils import CdtDataLoader
from omegaconf import DictConfig

from torch.nn import Sequential

from typing_extensions import override
import attr

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.transforms.spectrogram import Spectrogram
from src.transforms.frame import Frame

@attr.define(kw_only=True)
class EcoacousticsDataModuleEdit(EcoacousticsDataModule):

    @override
    def prepare_data(self,
                     *args: Any,
                     **kwargs: Any) -> None:

        # download the data
        data = Ecoacoustics(
            root=self.root,
            download=False,
            segment_len=self.segment_len,
            target_attrs=self.target_attrs,
        )

    '''

    def train_dataloader(self, **kwargs) -> CdtDataLoader:
        pass

    def test_dataloader(self, **kwargs) -> CdtDataLoader:
        pass

    def val_dataloader(self, **kwargs) -> CdtDataLoader:
        val_loader = self.make 
    '''

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint"""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = EcoacousticsDataModuleEdit()

