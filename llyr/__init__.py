import os
from pathlib import Path
import multiprocessing as mp
import glob

import numpy as np
import zarr
import h5py

from .plot import Plot
from .calc import Calc


def op(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Wrong path.")
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
                del self[dset]
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


def merge_table(m):
    for d in ["m", "B_ext"]:
        if f"table/{d}x" in m:
            x = m[f"table/{d}x"]
            y = m[f"table/{d}y"]
            z = m[f"table/{d}z"]
            m.create_dataset(f"table/{d}", data=np.array([x, y, z]).T)
            del m[f"table/{d}x"]
            del m[f"table/{d}y"]
            del m[f"table/{d}z"]


def h5_to_zarr(p, remove=False):
    source = h5py.File(p, "r")
    dest = zarr.open(p.replace(".h5", ".zarr"), mode="a")
    print("Copying:", p)
    zarr.copy_all(source, dest)
    print("Merging tables ..")
    merge_table(dest)
    source.close()
    print("Removing ...")
    if remove:
        os.remove(p)
    print("Done")


def load_ovf(path: str):
    with open(path, "rb") as f:
        dims = np.array([0, 0, 0, 0])
        while True:
            line = f.readline().strip().decode("ASCII")
            if "valuedim" in line:
                dims[3] = int(line.split(" ")[-1])
            if "xnodes" in line:
                dims[2] = int(line.split(" ")[-1])
            if "ynodes" in line:
                dims[1] = int(line.split(" ")[-1])
            if "znodes" in line:
                dims[0] = int(line.split(" ")[-1])
            if "Begin: Data" in line:
                break
        count = int(dims[0] * dims[1] * dims[2] * dims[3] + 1)
        arr = np.fromfile(f, "<f4", count=count)[1:].reshape(dims)
    return arr


def ovf_to_zarr(path, prefixes=("m", "stable")):
    m = zarr.open(path.replace(".out", ".zarr"), "w")

    for prefix in prefixes:
        print(f"{path}/{prefix}*.ovf")
        paths = sorted(glob.glob(f"{path}/{prefix}*.ovf"))
        if len(paths) > 0:
            s = load_ovf(paths[0]).shape
            dset = m.create_dataset(prefix, shape=((len(paths),) + s), dtype=np.float32)
            pool = mp.Pool(processes=int(mp.cpu_count() - 1))
            for i, d in enumerate(pool.imap(load_ovf, paths)):
                dset[i] = d
