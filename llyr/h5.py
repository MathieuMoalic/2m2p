from typing import Tuple, Union, Optional
import os
import multiprocessing as mp

import numpy as np
from tqdm.notebook import tqdm
import h5py

from .ovf import save_ovf, load_ovf


class H5:
    def __init__(self, llyr):
        self.llyr = llyr

    def save_as_ovf(self, arr: np.ndarray, name: str):
        path = self.llyr.path.replace(f"{self.llyr.name}", f"{name}.ovf")
        save_ovf(path, arr, self.llyr.dx, self.llyr.dy, self.llyr.dz)

    def create_h5(self, force: bool) -> bool:
        """Creates an empty .h5 file"""
        if force:
            with h5py.File(self.llyr.path, "w"):
                return True
        else:
            if os.path.isfile(self.llyr.path):
                input_string: str = input(
                    f"{self.llyr.path} already exists, overwrite it [y/n]?"
                )
                if input_string.lower() in ["y", "yes"]:
                    with h5py.File(self.llyr.path, "w"):
                        return True
        return False

    def shape(self, dset: str) -> Tuple:
        with h5py.File(self.llyr.path, "r") as f:
            return f[dset].shape

    def delete(self, dset: str) -> None:
        """deletes dataset"""
        with h5py.File(self.llyr.path, "a") as f:
            del f[dset]

    def move(self, source: str, destination: str) -> None:
        """move dataset or attribute"""
        with h5py.File(self.llyr.path, "a") as f:
            f.move(source, destination)

    def add_attr(
        self,
        key: str,
        val: Union[str, int, float, slice, Tuple[Union[int, slice], ...]],
        dset: Optional[str] = None,
    ) -> None:
        """set a new attribute"""
        if dset is None:
            with h5py.File(self.llyr.path, "a") as f:
                f.attrs[key] = val
        else:
            with h5py.File(self.llyr.path, "a") as f:
                f[dset].attrs[key] = val

    def add_dset(self, arr: np.ndarray, name: str, force: bool = False):
        if name in self.llyr.dsets:
            if force:
                self.delete(name)
            else:
                raise NameError(
                    f"Dataset with name '{name}' already exists, you can use 'force=True'"
                )
        with h5py.File(self.llyr.path, "a") as f:
            f.create_dataset(name, data=arr)

    def get_dset(self, dset, slices):
        with h5py.File(self.llyr.path, "r") as f:
            return f[dset][slices]

    def get_attrs(self, dset):
        with h5py.File(self.llyr.path, "r") as f:
            return dict(f[dset].attrs)

    def load_dset(self, name: str, dset_shape: tuple, ovf_paths: list) -> None:
        with h5py.File(self.llyr.path, "a") as f:
            dset = f.create_dataset(name, dset_shape, np.float32)
            with mp.Pool(processes=int(mp.cpu_count() - 1)) as p:
                for i, data in enumerate(
                    tqdm(
                        p.imap(load_ovf, ovf_paths),
                        leave=False,
                        desc=name,
                        total=len(ovf_paths),
                    )
                ):
                    dset[i] = data
