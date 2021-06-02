from typing import Union, Tuple, Optional
import os
import multiprocessing as mp

from tqdm import tqdm
import h5py
import numpy as np

from ._ovf import load_ovf


class H5:
    def __init__(self, path):
        path = os.path.abspath(path)
        if path[-3:] != ".h5":
            self.path = f"{path}.h5"
        else:
            self.path = path

    def create_h5(self, override: bool) -> bool:
        """Creates an empty .h5 file"""
        if override:
            with h5py.File(self.path, "w"):
                return True
        else:
            if os.path.isfile(self.path):
                input_string: str = input(
                    f"{self.path} already exists, overwrite it [y/n]?"
                )
                if input_string.lower() in ["y", "yes"]:
                    with h5py.File(self.path, "w"):
                        return True
        return False

    @property
    def dsets(self) -> dict:
        dsets = {}
        with h5py.File(self.path, "r") as f:
            for key in f.keys():
                dsets[key] = f[key].shape
        return dsets

    @property
    def attrs(self) -> dict:
        attrs = {}
        with h5py.File(self.path, "r") as f:
            for k, v in f.attrs.items():
                attrs[k] = v
        return attrs

    @property
    def root(self) -> dict:
        attrs = {}
        with h5py.File(self.path, "r") as f:
            for k, v in f.items():
                attrs[k] = v
        return attrs

    def shape(self, dset: str) -> Tuple:
        with h5py.File(self.path, "r") as f:
            return f[dset].shape

    def delete(self, dset: str) -> None:
        """deletes dataset"""
        with h5py.File(self.path, "a") as f:
            del f[dset]

    def move(self, source: str, destination: str) -> None:
        """move dataset or attribute"""
        with h5py.File(self.path, "a") as f:
            f.move(source, destination)

    def add_attr(
        self,
        key: str,
        val: Union[str, int, float, slice, Tuple[Union[int, slice], ...]],
        dset: Optional[str] = None,
    ) -> None:
        """set a new attribute"""
        if dset is None:
            with h5py.File(self.path, "a") as f:
                f.attrs[key] = val
        else:
            with h5py.File(self.path, "a") as f:
                f[dset].attrs[key] = val

    def add_dset(self, arr: np.ndarray, name: str, override: bool = False):
        if name in self.dsets:
            if override:
                self.delete(name)
            else:
                raise NameError(
                    f"Dataset with name '{name}' already exists, you can use 'override=True'"
                )
        with h5py.File(self.path, "a") as f:
            f.create_dataset(name, data=arr)

    def get_dset(self, dset, slices):
        with h5py.File(self.path, "r") as f:
            return f[dset][slices]

    def load_dset(self, name: str, dset_shape: tuple, ovf_paths: list) -> None:
        with h5py.File(self.path, "a") as f:
            dset = f.create_dataset(name, dset_shape, np.float32)
            with mp.Pool(processes=int(mp.cpu_count())) as p:
                for i, data in enumerate(
                    tqdm(
                        p.imap(load_ovf, ovf_paths),
                        leave=False,
                        desc=name,
                        total=len(ovf_paths),
                    )
                ):
                    dset[i] = data
