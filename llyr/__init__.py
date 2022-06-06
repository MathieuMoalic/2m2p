import os
from pathlib import Path
import shutil

import numpy as np
import zarr
from .plot import Plot
from .calc import Calc

from ._utils import (
    h5_to_zarr,
    load_ovf,
    merge_table,
    get_ovf_parms,
    out_to_zarr,
    hsl2rgb,
    MidpointNormalize,
    save_ovf,
    get_cmaps,
    add_radial_phase_colormap,
)
from ._iplot import iplotp
from ._iplot2 import iplotp2


__all__ = [
    "h5_to_zarr",
    "load_ovf",
    "merge_table",
    "get_ovf_parms",
    "out_to_zarr",
    "hsl2rgb",
    "iplot",
    "MidpointNormalize",
    "save_ovf",
    "get_cmaps",
    "add_radial_phase_colormap",
]


def iplot(*args, **kwargs):
    return iplotp(op, *args, **kwargs)


def iplot2(*args, **kwargs):
    return iplotp2(op, *args, **kwargs)


def op(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path Not Found : '{path}'")
    if "ssh://" in path:
        return Group(zarr.storage.FSStore(path))
    else:
        return Group(zarr.storage.DirectoryStore(path))


class Group(zarr.hierarchy.Group):
    def __init__(self, store) -> None:
        zarr.hierarchy.Group.__init__(self, store)
        self.abs_path = Path(store.path).absolute()
        self.sim_name = self.abs_path.name.replace(self.abs_path.suffix, "")
        self.plot = Plot(self)
        self.calc = Calc(self)
        self.reload()

    def __repr__(self) -> str:
        return f"Llyr('{self.sim_name}')"

    def __str__(self) -> str:
        return f"Llyr('{self.sim_name}')"

    def reload(self):
        self._update_class_dict()

    def _update_class_dict(self):
        for k, v in self.attrs.items():
            self.__dict__[k] = v

    def rm(self, dset: str):
        shutil.rmtree(f"{self.abs_path}/{dset}", ignore_errors=True)

    @property
    def pp(self):
        return self.tree(expand=True)

    @property
    def p(self):
        print(self.tree())

    @property
    def snap(self):
        self.plot.snapshot_png("stable")

    def c_to_comp(self, c):
        return ["mx", "my", "mz"][c]

    def comp_to_c(self, c):
        return {"mx": 0, "my": 1, "mz": 2}[c]

    def get_mode(self, dset: str, f: float, c: int = None):
        if f"modes/{dset}/arr" not in self:
            print("Calculating modes ...")
            self.calc.modes(dset)
        fi = int((np.abs(self[f"modes/{dset}/freqs"][:] - f)).argmin())
        arr = self[f"modes/{dset}/arr"][fi]
        if c is None:
            return arr
        else:
            return arr[..., c]

    def check_path(self, dset: str, force: bool = False):
        if dset in self:
            if force:
                self.rm(dset)
            else:
                raise NameError(
                    f"The dataset:'{dset}' already exists, you can use 'force=True'"
                )

    def make_report(self, dset="m"):
        os.makedirs(f"{self.apath}/report")
        r = self.plot.report(dset=dset, save=f"{self.apath}/report/spectra.pdf")
        for peak in r.peaks:
            self.plot.anim(
                dset=dset,
                f=peak.freq,
                save_path=f"{self.apath}/report/{peak.freq:.2f}.gif",
            )
