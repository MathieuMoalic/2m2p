import h5py  # type: ignore
import numpy as np
from typing import *

from m2p2._make import Make
from m2p2._transform import Transform


class H5(Make, Transform):
    def __init__(self, path: str, loadpath: Optional[str] = None):
        self.path = path
        self.name = path.split("/")[-1]
        if loadpath is not None:
            self.make(loadpath)
        self.getitem_dset: Optional[str] = None

    def __getitem__(
        self,
        index: Union[str, Tuple[Union[int, slice], ...]],
    ) -> Union["H5", float, np.ndarray]:
        with h5py.File(self.path, "r") as f:

            # if slicing
            if isinstance(index, (slice, tuple, int)):
                # if dset is defined
                if isinstance(self.getitem_dset, str):
                    out_dset: np.ndarray = f[self.getitem_dset][index]
                    self.getitem_dset = None
                    return out_dset
                else:
                    raise AttributeError("You can only slice datasets")

            elif isinstance(index, str):
                # if dataset
                if index in list(f.keys()):
                    self.getitem_dset = index
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
        with h5py.File(self.path, "a") as f:
            f.create_dataset(name, data=data)

    def a(
        self,
        name: str,
        key: str,
        val: Union[str, int, float, slice, Tuple[Union[int, slice], ...]],
    ) -> None:
        with h5py.File(self.path, "a") as f:
            f[name].attrs[key] = val

    @property
    def p(self) -> None:
        with h5py.File(self.path, "r") as f:
            print("Datasets:")
            for key in list(f.keys()):
                print(f"    {key:<15}: {f[key].shape}")
            print("Attributes:")
            for key in list(f.attrs.keys()):
                if key != "script":
                    print(f"    {key:<15}= {f.attrs[key]}")

    def freqs(self) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            dt: float = f.attrs["dt"]
            tshape: int = f["WG"].shape[0]
            out: np.ndarray = np.fft.rfftfreq(tshape, dt * 1e9)
        return out

    def kvecs(self) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            dx: float = f.attrs["dx"]
            xshape: int = f["WG"].shape[3]
            out: np.ndarray = (
                np.fft.fftshift(np.fft.fftfreq(xshape, dx) * 2 * np.pi) / 1e6
            )
        return out

    def d(self, dset: str) -> None:
        with h5py.File(self.path, "a") as f:
            del f[dset]

    def move(self, source: str, destination: str) -> None:
        with h5py.File(self.path, "a") as f:
            f.move(source, destination)
