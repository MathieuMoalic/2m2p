
from ..base import Base

from .._utils import save_ovf


class ovf_anim(Base):
    def plot(
        self,
        savepath: str = None,
        dset: str = "m",
        slices = (slice(None),slice(None),slice(None,None,5),slice(None,None,5),slice(None)),
    ):
        if savepath is None:
            savepath = f"anim/{dset}"
        self.m.rm(savepath)
        self.m.mkdir(savepath)
        arr = self.m[dset][slices]
        print(f"Shape : {arr.shape}")
        for i,a in enumerate(arr):
            save_ovf(
                f"{self.m.abs_path}/{savepath}/{i}.ovf",
                a,
                self.m.dx,
                self.m.dy,
                self.m.dz,
            )
        print(f"Saved in: {savepath}")
