from typing import Optional, Union, Tuple

import numpy as np

from ._plot import Plot
from ._h5 import H5
from ._make import Make
from ._ovf import save_ovf


class Llyr:
    def __init__(self, h5_path: str) -> None:
        self.h5 = H5(h5_path)
        self.name = h5_path.split("/")[-1].replace(".h5", "")
        self._getitem_dset: Optional[str] = None
        self.plot = Plot(self)

    def make(
        self,
        load_path: Optional[str] = None,
        tmax=None,
        override=False,
        delete_out=False,
    ):
        Make(self.h5, load_path, tmax, override, delete_out)
        return self

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
            if index in self.h5.dsets:
                self._getitem_dset = index
                return self
            # if attribute
            elif index in self.h5.attrs:
                out_attribute: float = self.h5.attrs[index]
                return out_attribute
            else:
                raise KeyError("No such Dataset or Attribute")
        else:
            raise TypeError()

    @property
    def dsets(self) -> dict:
        return self.h5.dsets

    @property
    def attrs(self) -> dict:
        return self.h5.attrs

    @property
    def mx3(self) -> str:
        print(self["mx3"])

    @property
    def dt(self) -> float:
        return self.h5.attrs["dt"]

    @property
    def dx(self) -> float:
        return self.h5.attrs["dx"]

    @property
    def dy(self) -> float:
        return self.h5.attrs["dy"]

    @property
    def dz(self) -> float:
        return self.h5.attrs["dz"]

    @property
    def p(self) -> None:
        print("Datasets:")
        for dset_name, dset_shape in self.h5.dsets.items():
            print(f"    {dset_name:<15}: {dset_shape}")
        print("Global Attributes:")
        for key, val in self.h5.attrs.items():
            if key in ["mx3", "script"]:
                val = val.replace("\n", "")
                print(f"    {key:<15}= {val[:10]}...")
            else:
                print(f"    {key:<15}= {val}")

    def save_as_ovf(self, arr: np.ndarray, name: str):
        path = self.h5.path.replace(f"{self.name}", f"{name}.ovf")
        save_ovf(path, arr, self.dx, self.dy, self.dz)
