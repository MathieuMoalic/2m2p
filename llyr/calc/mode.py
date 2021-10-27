import numpy as np
from ..base import Base


class mode(Base):
    def calc(self, dset: str, f: float, c: int):
        if f"modes/{dset}/arr" not in self.llyr.dsets:
            print("Calculating modes ...")
            self.llyr.calculate_modes(dset)
        fi = int((np.abs(self.llyr[f"modes/{dset}/freqs"][:] - f)).min())
        _mode = self.llyr[f"modes/{dset}/arr"][fi][..., c]
        return _mode
