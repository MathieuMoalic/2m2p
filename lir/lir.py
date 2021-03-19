import h5py  # type: ignore
import numpy as np
from typing import *
import struct

from lir._make import Make
from lir._transform import Transform
from lir._ovf import Ovf


class Lir(Make, Transform, Ovf):
    def __init__(self, h5_path: str, load_path: Optional[str] = None, tmax=None, force=False):
        self.force = force
        self.h5_path = h5_path
        self.name = h5_path.split("/")[-1][:-3]
        if load_path is not None:
            self.make(load_path,tmax=tmax)
        self._getitem_dset: Optional[str] = None

    def __getitem__(
        self,
        index: Union[str, Tuple[Union[int, slice], ...]],
    ) -> Union["H5", float, np.ndarray]:
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

    def dset(self, name: str, data: np.ndarray) -> None:
        with h5py.File(self.h5_path, "a") as f:
            f.create_dataset(name, data=data)

    def a(
        self,
        name: str,
        key: str,
        val: Union[str, int, float, slice, Tuple[Union[int, slice], ...]],
    ) -> None:
        with h5py.File(self.h5_path, "a") as f:
            f[name].attrs[key] = val

    def t(self,script_name,ovf_folder="/mnt/g/Mathieu/simulations/stable",dset="stable",t=0):
        linux_ovf_name = f"{ovf_folder}/{self.name}.ovf"
        windows_ovf_name = f"G:{ovf_folder[6:]}/{self.name}.ovf"
        script_lines = self["mx3"].split("\n")
        load_line = f'm.loadfile("{windows_ovf_name}")\n'

        with open(script_name,"w") as f:
            prefix = ""
            for i,line in enumerate(script_lines):
                f.write(prefix+line+"\n")
                if "dind.setregio" in line:
                    f.write("\n")
                    f.write(load_line)
                    prefix = "// "

        self.save_ovf(dset,linux_ovf_name,t=t)

    def mx3(self,savepath=None):
        if savepath is None:
            print(self["mx3"])
        else:
            with open(savepath,"w") as f:
                f.writelines(self["mx3"])

    @property
    def p(self) -> None:
        with h5py.File(self.h5_path, "r") as f:
            print("Datasets:")
            for key in list(f.keys()):
                print(f"    {key:<15}: {f[key].shape}")
            print("Attributes:")
            for key in list(f.attrs.keys()):
                if key != "mx3":
                    print(f"    {key:<15}= {f.attrs[key]}")
                else:
                    print(f"    {key:<15}= {f.attrs[key][:10]}...")

    def freqs(self) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as f:
            dt: float = f.attrs["dt"]
            tshape: int = f["WG"].shape[0]
            out: np.ndarray = np.fft.rfftfreq(tshape, dt * 1e9)
        return out

    def kvecs(self) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as f:
            dx: float = f.attrs["dx"]
            xshape: int = f["WG"].shape[3]
            out: np.ndarray = (
                np.fft.fftshift(np.fft.fftfreq(xshape, dx) * 2 * np.pi) / 1e6
            )
        return out

    def d(self, dset: str) -> None:
        with h5py.File(self.h5_path, "a") as f:
            del f[dset]

    def move(self, source: str, destination: str) -> None:
        with h5py.File(self.h5_path, "a") as f:
            f.move(source, destination)
