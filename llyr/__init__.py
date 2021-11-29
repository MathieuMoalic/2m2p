from typing import Optional, Tuple, Union, Dict
import os
from pathlib import Path

import numpy as np
import zarr

from .plot import Plot
from .calc import Calc
from .make import add_table


def op(path):
    if ".zarr" not in path or ".out" not in path:
        path = f"{path}.zarr"
    if "ssh://" in path:
        return Group(zarr.storage.FSStore(path))
    else:
        return Group(zarr.storage.DirectoryStore(path))


class Group(zarr.hierarchy.Group):
    def __init__(self, path: str) -> None:
        zarr.hierarchy.Group.__init__(self, path)
        self.apath = Path(path.path).absolute()
        self.aname = self.apath.name.replace(self.apath.suffix, "")
        self.plot = Plot(self)
        self.calc = Calc(self)
        self.reload()

    def __repr__(self) -> str:
        return f"Llyr('{self.aname}')"

    def __str__(self) -> str:
        return f"Llyr('{self.aname}')"

    def reload(self):
        add_table(self)
        self._update_class_dict()

    def _update_class_dict(self):
        for k, v in self.attrs.items():
            self.__dict__[k] = v

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

    def modes(self, dset: str, f: float, c: int = None):
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
                del self[dset]
            else:
                raise NameError(
                    f"The dataset:'{dset}' already exists, you can use 'force=True'"
                )

    def make_report(self):
        os.makedirs(f"{self.apath}/report")
        r = self.plot.report(save=f"{self.apath}/report/spectra.pdf")
        for peak in r.peaks:
            self.plot.anim(
                f=peak.freq, save_path=f"{self.apath}/report/{peak.freq:.2f}.gif"
            )
