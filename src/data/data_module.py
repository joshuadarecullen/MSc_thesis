from typing import List, Tuple, Any, Union, Any, Dict
from conduit.data.datasets.audio.ecoacoustics import Ecoacoustics, SoundscapeAttr

class EcoacousticsDataModuleEdit(EcoacousticsDataModule):
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

    def train_dataloader(self, **kwargs) -> CdtDataLoader:
        pass

    def test_dataloader(self, **kwargs) -> CdtDataLoader:
        pass

    def val_dataloader(self, **kwargs) -> CdtDataLoader:
        pass

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

