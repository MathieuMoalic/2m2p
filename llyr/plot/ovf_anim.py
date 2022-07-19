import os

import numpy as np

from ..base import Base

from .._utils import save_ovf


class ovf_anim(Base):
    def plot(
        self,
        savepath,
        dset: str = "m",
        f: float = 9,
        step: int = 10,
        periods: int = 1,
        repeat: int = 1,
        p=0,
    ):
        arr = self.m.calc.anim(dset, f, periods=periods)[:, :, ::step, ::step]
        arr = self.m.stable[:, :, ::step, ::step] * p + arr * (1 - p)
        arr = np.tile(arr, (1, 1, repeat, repeat, 1))
        arr = np.ma.masked_equal(arr, 0)
        arr /= 10
        self.m.rm(f"anim/{f}")
        self.m.create_dataset(f"anim/{f}", shape=arr.shape, dtype=np.float32)
        # name = f"{savepath}/{}"
        os.makedirs(savepath, exist_ok=True)
        for t in range(arr.shape[0]):
            save_ovf(
                f"{savepath}/{t}.ovf",
                arr[t],
                self.m.dx,
                self.m.dy,
                self.m.dz,
            )
        print(f"Saved in: {savepath}")
