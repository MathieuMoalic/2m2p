from typing import Optional, Tuple, Union, Dict
import os
from pathlib import Path

import numpy as np
import h5py

from .plot import Plot
from .calc import Calc
from .h5 import H5
from .make import Make
from .ovf import save_ovf
from ._utils import cspectra_b


class Llyr:
    def __init__(self, path: str) -> None:
        path = os.path.abspath(path)
        self._add_path(path)
        self._getitem_dset: Optional[str] = None
        self.plot = Plot(self)
        self.calc = Calc(self)
        self.h5 = H5(self)
        self.make = Make(self).make

    def _add_path(self, path):
        p = Path(path)
        if p.suffix == "":
            self.name = p.name
        else:
            self.name = p.name.replace(p.suffix, "")
        self.path = f"{str(p.parent)}/{self.name}.h5"

    def __repr__(self) -> str:
        return f"Llyr('{self.name}')"

    def __str__(self) -> str:
        return f"Llyr('{self.name}')"

    def __getitem__(
        self,
        index: Union[str, Tuple[Union[int, slice], ...]],
    ) -> Union["Llyr", float, np.ndarray]:
        if isinstance(index, (slice, tuple, int)):
            # if dset is defined
            if isinstance(self._getitem_dset, str):
                out_dset: np.ndarray = self.h5.get_dset(self._getitem_dset, index)
                self._getitem_dset = None
                return out_dset
            else:
                raise AttributeError("You can only slice datasets")

        elif isinstance(index, str):
            # if dataset
            if index in self.dsets:
                self._getitem_dset = index
                return self
            # if attribute
            elif index in self.attrs:
                out_attribute: float = self.attrs[index]
                return out_attribute
            else:
                raise KeyError("No such Dataset or Attribute")
        else:
            print()
            raise TypeError(f"{index}, {type(index)}")

    @property
    def mx3(self) -> None:
        print(self["mx3"])

    @property
    def dt(self) -> float:
        return self.attrs["dt"]

    @property
    def dx(self) -> float:
        return self.attrs["dx"]

    @property
    def dy(self) -> float:
        return self.attrs["dy"]

    @property
    def dz(self) -> float:
        return self.attrs["dz"]

    @property
    def pp(self) -> None:
        print("Datasets:")
        for dset_name, dset_shape in self.dsets.items():
            print(f"    {dset_name:<20}: {dset_shape}")
        print("Global Attributes:")
        for key, val in self.attrs.items():
            if key in ["mx3", "script", "logs"]:
                val = val.replace("\n", "")
                print(f"    {key:<20}= {val[:10]}...")
            else:
                print(f"    {key:<20}= {val}")

    @property
    def dsets(self) -> Dict[str, str]:
        def add_dset(name, obj):
            # pylint: disable=protected-access
            if isinstance(obj, h5py._hl.dataset.Dataset):
                dsets[name] = obj.shape

        dsets: Dict[str, str] = {}
        with h5py.File(self.path, "r") as f:
            f.visititems(add_dset)
        return dsets

    @property
    def attrs(self) -> dict:
        attrs = {}
        with h5py.File(self.path, "r") as f:
            for k, v in f.attrs.items():
                attrs[k] = v
        return attrs

    @property
    def snap(self):
        self.plot.snapshot_png("stable")

    def modes(self, dset: str, f: float, c: int = None):
        if f"modes/{dset}/arr" not in self.dsets:
            print("Calculating modes ...")
            self.calc.modes(dset)
        fi = int((np.abs(self[f"modes/{dset}/freqs"][:] - f)).argmin())
        arr = self[f"modes/{dset}/arr"][fi]
        if c is None:
            return arr
        else:
            return arr[..., c]

    def check_path(self, name: str, override: bool = False):
        if name in self.dsets and not override:
            raise NameError(
                f"The dataset:'{name}' already exists, you can use 'override=True'"
            )


cspectra = cspectra_b(Llyr)
