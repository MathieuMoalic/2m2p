from typing import Optional, Union, Tuple
import os

import h5py
import numpy as np

from ._make import Make
from ._ovf import save_ovf, load_ovf, get_ovf_parms
from ._plot import imshow
from ._disp import Disp


class Llyr:
    def __init__(
        self, h5_path: str, load_path: Optional[str] = None, tmax=None, force=False
    ) -> None:
        self.h5_path = self._clean_path(h5_path)
        self.name = self.h5_path.split("/")[-1][:-3]
        if load_path is not None:
            Make(self, load_path, force=force, tmax=tmax)
        self._getitem_dset: Optional[str] = None

    def __repr__(self):
        return f"Llyr('{self.h5_path}')"

    def __str__(self):
        return f"Llyr('{self.h5_path}')"

    def _clean_path(self, path):
        path = os.path.abspath(path)
        if path[-3:] != ".h5":
            out_path = f"{path}.h5"
        else:
            out_path = path
        return out_path

    def __getitem__(
        self,
        index: Union[str, Tuple[Union[int, slice], ...]],
    ) -> Union["Llyr", float, np.ndarray]:
        with h5py.File(self.h5_path, "r") as f:

            # if slicing

            if isinstance(index, (slice, tuple, int)):
                # if dset is defined
                if isinstance(self._getitem_dset, str):
                    out_dset: np.ndarray = f[self._getitem_dset][index]
                    self._getitem_dset = None
                    return out_dset
                else:
                    raise AttributeError("You can only slice datasets")

            elif isinstance(index, str):
                # if dataset
                if index in list(f.keys()):
                    self._getitem_dset = index
                    return self
                # if attribute
                elif index in list(f.attrs.keys()):
                    out_attribute: float = f.attrs[index]
                    return out_attribute
                else:
                    raise KeyError("No such Dataset or Attribute")
            else:
                raise TypeError()

    def set_attr(
        self,
        name: str,
        key: str,
        val: Union[str, int, float, slice, Tuple[Union[int, slice], ...]],
    ) -> None:
        """set a new attribute"""
        with h5py.File(self.h5_path, "a") as f:
            f[name].attrs[key] = val

    def shape(self, dset: str) -> (int, int, int, int, int):
        with h5py.File(self.h5_path, "r") as f:
            return f[dset].shape

    def save_mx3(self, savepath: str) -> None:
        """prints or saves the mx3"""
        with open(savepath, "w") as f:
            f.writelines(self["mx3"])

    def save_ovf(self, ovf_path: str, dset: str, t: int = 0) -> None:
        arr = np.expand_dims(self[dset][t], axis=0)
        save_ovf(ovf_path, arr, self.dx, self.dy, self.dz)

    def imshow(self, dset, t=-1, zero=True):
        imshow(self, dset, t=t, zero=zero)

    @staticmethod
    def load_ovf(ovf_path: str):
        return load_ovf(ovf_path)

    @property
    def mx3(self) -> str:
        return self["mx3"]

    @property
    def dt(self) -> float:
        return self["dt"]

    @property
    def dx(self) -> float:
        return self["dx"]

    @property
    def dy(self) -> float:
        return self["dy"]

    @property
    def dz(self) -> float:
        return self["dz"]

    @property
    def p(self) -> None:
        with h5py.File(self.h5_path, "r") as f:
            print("Datasets:")
            for key, val in f.items():
                print(f"    {key:<15}: {val.shape}")
                if f[key].attrs:
                    print(f"    Attributes of {key}:")
                for akey, aval in f[key].attrs.items():
                    if isinstance(aval, np.ndarray):
                        aval = f"{aval.shape} : min={aval.min()}, max={aval.max()}"
                    print(f"        {akey:<11}= {aval}")

            print("Global Attributes:")
            for key, val in f.attrs.items():
                if key in ["mx3", "script"]:
                    val = val.replace("\n", "")
                    print(f"    {key:<15}= {val[:10]}...")
                else:
                    print(f"    {key:<15}= {val}")

    @property
    def dsets(self) -> list:
        with h5py.File(self.h5_path, "r") as f:
            dsets = list(f.keys())
        return dsets

    @property
    def attrs(self) -> list:
        with h5py.File(self.h5_path, "r") as f:
            attrs = list(f.attrs.keys())
        return attrs

    def delete(self, dset: str) -> None:
        """deletes dataset"""
        with h5py.File(self.h5_path, "a") as f:
            del f[dset]

    def move(self, source: str, destination: str) -> None:
        """move dataset or attribute"""
        with h5py.File(self.h5_path, "a") as f:
            f.move(source, destination)
