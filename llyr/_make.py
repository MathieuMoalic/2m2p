from typing import Tuple, List, Optional
from glob import glob
import os
import multiprocessing as mp

import h5py
import numpy as np
from tqdm import tqdm

from ._ovf import load_ovf, get_ovf_parms


class Make:
    def __init__(self, h5, load_path, force=False, tmax=None):
        self.h5 = h5
        self.make(load_path, force, tmax)

    def add_table(self, table_path: str, dset_name: str = "table") -> None:
        """Adds a the mumax table.txt file as a dataset"""
        if os.path.isfile(table_path):
            with open(table_path, "r") as table:
                header: str = table.readline()
                data: np.ndarray = np.loadtxt(table).T
                dt: float = (data[0, -1] - data[0, 0]) / (data.shape[1] - 1)
            with h5py.File(self.h5.h5_path, "a") as f:
                tableds = f.create_dataset(dset_name, data=data)
                tableds.attrs["header"] = header
                f.attrs["dt"] = dt
        else:
            print("table.txt not found")

    def _get_paths(self, load_path: str) -> Tuple[str, str]:
        """Cleans the input string and return the path for .out folder and .mx3 file"""
        load_path = load_path.replace(".mx3", "").replace(".out", "").replace(".h5", "")
        out_path = f"{load_path}.out"
        mx3_path = f"{load_path}.mx3"
        return out_path, mx3_path

    def add_mx3(self, mx3_path: str) -> None:
        """Adds the mx3 file to the f.attrs"""
        if os.path.isfile(mx3_path):
            with open(mx3_path, "r") as mx3:
                with h5py.File(self.h5.h5_path, "a") as f:
                    f.attrs["mx3"] = mx3.read()
        else:
            print(f"{mx3_path} not found")

    def _create_h5(self, force) -> bool:
        """Creates an empty .h5 file"""
        if force:
            with h5py.File(self.h5.h5_path, "w"):
                return True
        else:
            if os.path.isfile(self.h5.h5_path):
                input_string: str = input(
                    f"{self.h5.h5_path} already exists, overwrite it [y/n]?"
                )
                if input_string.lower() in ["y", "yes"]:
                    with h5py.File(self.h5.h5_path, "w"):
                        return True
        return False

    def _get_dset_prefixes(self, out_path: str) -> List[str]:
        """From the .out folder, get the list of prefixes, each will correspond to a different dataset"""
        prefixes = [
            i.split("/")[-1].replace("000000.ovf", "")
            for i in glob(f"{out_path}/*00000.ovf")
        ]
        if os.path.isfile(f"{out_path}/stable.ovf"):
            prefixes.append("stable")
        return prefixes

    def _get_dset_name(self, prefix: str) -> str:
        """From the prefix, this tries to return a human readable version"""
        common_prefix_to_name = (
            ("m_xrange", "ND"),
            ("m_zrange", "WG"),
            ("B_demag_xrange", "ND_B"),
            ("B_demag_zrange", "WG_B"),
            ("stable", "stable"),
        )
        for i in common_prefix_to_name:
            if i[0] in prefix:
                return i[1]
        return prefix

    def _save_stepsize(self, parms: dict) -> None:
        for key in ["dx", "dy", "dz"]:
            if key not in self.h5.attrs:
                with h5py.File(self.h5.h5_path, "a") as f:
                    f.attrs[key] = parms[key]

    def add_dset(
        self,
        out_path: str,
        prefix: str,
        name: Optional[str] = None,
        tmax: Optional[int] = None,
        force: bool = False,
    ) -> None:
        """Creates a dataset from an input .out folder path and a prefix (i.e. "m00")"""
        ovf_paths = sorted(glob(f"{out_path}/{prefix}*.ovf"))[:tmax]
        # load one file to initialize the h5 dataset with the correct shape
        ovf_parms = get_ovf_parms(ovf_paths[0])
        self._save_stepsize(ovf_parms)
        ovf_shape = ovf_parms["shape"]
        dset_shape = (len(ovf_paths),) + ovf_shape
        if name is None:
            name = self._get_dset_name(prefix)

        with h5py.File(self.h5.h5_path, "a") as f:
            if force and name in list(f.keys()):
                del f[name]
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

    def add_np_dset(self, arr: np.ndarray, name: str, force: bool = False):
        with h5py.File(self.h5.h5_path, "a") as f:
            if name in self.h5.dsets():
                if force:
                    del f[name]
                    f.create_dataset(name, data=arr)
                else:
                    raise NameError(
                        "A dataset with this name already exists, use force=True to override it"
                    )
            else:
                f.create_dataset(name, data=arr)

    def make(self, load_path: str, force: bool, tmax: Optional[int]) -> None:
        """Automatically parse the load_path and will create datasets"""
        self._create_h5(force)
        out_path, mx3_path = self._get_paths(load_path)
        self.add_table(f"{out_path}/table.txt")
        self.add_mx3(mx3_path)
        dset_prefixes = self._get_dset_prefixes(out_path)
        for dset_prefix in dset_prefixes:
            self.add_dset(out_path, dset_prefix, tmax=tmax)
