from typing import Optional, Union, Tuple, List
import struct
import os
from glob import glob
import multiprocessing as mp

import psutil
import h5py
import numpy as np
from tqdm.notebook import tqdm
import dask.array as da
import matplotlib.pyplot as plt
from dask.distributed import Client


class Lir:
    def __init__(
        self, h5_path: str, load_path: Optional[str] = None, tmax=None, force=False
    ) -> None:
        self.force = force

        self.h5_path = self._clean_path(h5_path)
        self.name = self.h5_path.split("/")[-1][:-3]
        if load_path is not None:
            self.make(load_path, tmax=tmax)
        self._getitem_dset: Optional[str] = None

    def __repr__(self):
        return f"Lir({self.h5_path})"

    def __str__(self):
        return f"Lir({self.h5_path})"

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
    ) -> Union["Lir", float, np.ndarray]:
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

    def shape(self, dset: str):
        with h5py.File(self.h5_path, "r") as f:
            return f[dset].shape

    def t(
        self,
        script_name: str,
        ovf_folder: str = "/mnt/g/Mathieu/simulations/stable",
        dset: str = "stable",
        t: int = 0,
    ) -> None:
        """Writes a new mx3 from the one saved in this h5 file, it will add the load line too"""
        linux_ovf_name = f"{ovf_folder}/{self.name}.ovf"
        windows_ovf_name = f"G:{ovf_folder[6:]}/{self.name}.ovf"
        script_lines = self["mx3"].split("\n")
        load_line = f'm.loadfile("{windows_ovf_name}")\n'

        with open(script_name, "w") as f:
            prefix = ""
            for line in script_lines:
                f.write(prefix + line + "\n")
                if "dind.setregio" in line:
                    f.write("\n")
                    f.write(load_line)
                    prefix = "// "

        self.save_ovf(dset, linux_ovf_name, t=t)

    def mx3(self, savepath: str = None) -> None:
        """prints or saves the mx3"""
        if savepath is None:
            print(self["mx3"])
        else:
            with open(savepath, "w") as f:
                f.writelines(self["mx3"])

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

    def dsets(self) -> list:
        with h5py.File(self.h5_path, "r") as f:
            dsets = list(f.keys())
        return dsets

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

    def _load_ovf(self, ovf_path: str, count: int, ovf_shape: Tuple(int)) -> np.ndarray:
        """Returns an np.ndarray from the ovf"""
        with open(ovf_path, "rb") as f:
            for _ in range(28):
                next(f)
            try:
                arr: np.ndarray = np.fromfile(f, "<f4", count=count)[1:].reshape(
                    ovf_shape
                )
            except ValueError as e:
                print("ovf files are corrupted")
                return e
        return arr

    def _getovf_parms(self, ovf_path: str) -> dict:
        """Return a tuple of the shape of the ovf file at the ovf_path"""
        with open(ovf_path, "rb") as f:
            while True:
                line = f.readline().strip().decode("ASCII")
                if "valuedim" in line:
                    c = int(line.split(" ")[-1])
                if "xnodes" in line:
                    x = int(line.split(" ")[-1])
                if "ynodes" in line:
                    y = int(line.split(" ")[-1])
                if "znodes" in line:
                    z = int(line.split(" ")[-1])
                if "xstepsize" in line:
                    dx = float(line.split(" ")[-1])
                if "ystepsize" in line:
                    dy = float(line.split(" ")[-1])
                if "zstepsize" in line:
                    dz = float(line.split(" ")[-1])
                # if "Desc: Total simulation time:" in line:
                #     dt = float(line.split("  ")[-2])
                if "End: Header" in line:
                    break
        parms = {"shape": (z, y, x, c), "dx": dx, "dy": dy, "dz": dz}
        return parms

    def add_table(self, table_path: str, dset_name: str = "table") -> None:
        """Adds a the mumax table.txt file as a dataset"""
        if os.path.isfile(table_path):
            with open(table_path, "r") as table:
                header: str = table.readline()
                data: np.ndarray = np.loadtxt(table).T
                dt: float = (data[0, -1] - data[0, 0]) / (data.shape[1] - 1)
            with h5py.File(self.h5_path, "a") as f:
                tableds = f.create_dataset(dset_name, data=data)
                tableds.attrs["header"] = header
                f.attrs["dt"] = dt
        else:
            print("table.txt not found")

    def _get_paths(self, load_path: str) -> Tuple[str, str]:
        """Cleans the input string and return the path for .out folder and .mx3 file"""
        if load_path[-3:] in ["mx3", "out"]:
            load_path = load_path[:-4]
        out_path = f"{load_path}.out"
        mx3_path = f"{load_path}.mx3"
        return out_path, mx3_path

    def add_mx3(self, mx3_path: str) -> None:
        """Adds the mx3 file to the f.attrs"""
        if os.path.isfile(mx3_path):
            with open(mx3_path, "r") as mx3:
                with h5py.File(self.h5_path, "a") as f:
                    f.attrs["mx3"] = mx3.read()
        else:
            print(f"{mx3_path} not found")

    def _create_h5(self) -> bool:
        """Creates an empty .h5 file"""
        if self.force:
            with h5py.File(self.h5_path, "w"):
                return True
        else:
            if os.path.isfile(self.h5_path):
                input_string: str = input(
                    f"{self.h5_path} already exists, overwrite it [y/n]?"
                )
                if input_string.lower() in ["y", "yes"]:
                    with h5py.File(self.h5_path, "w"):
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
            if key not in self.attrs():
                with h5py.File(self.h5_path, "a") as f:
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
        ovf_parms = self._getovf_parms(ovf_paths[0])
        self._save_stepsize(ovf_parms)
        ovf_shape = ovf_parms["shape"]
        dset_shape = (len(ovf_paths),) + ovf_shape
        # number of bytes in the data used in self._load_ovf, (+1 is for the security number of ovf)
        count = ovf_shape[0] * ovf_shape[1] * ovf_shape[2] * ovf_shape[3] + 1
        if name is None:
            name = self._get_dset_name(prefix)
        iterable = zip(
            ovf_paths, [ovf_shape] * len(ovf_paths), [count] * len(ovf_paths)
        )

        with h5py.File(self.h5_path, "a") as f:
            if force and name in list(f.keys()):
                del f[name]
            dset = f.create_dataset(name, dset_shape, np.float32)
            with mp.Pool(processes=int(mp.cpu_count())) as p:
                for i, data in enumerate(
                    tqdm(
                        p.starmap(self._load_ovf, iterable),
                        leave=False,
                        desc=name,
                        total=len(ovf_paths),
                    )
                ):
                    dset[i] = data

    def add_np_dset(self, arr: np.ndarray, name: str, force: bool = False):
        with h5py.File(self.h5_path, "a") as f:
            if name in self.dsets():
                if force:
                    del f[name]
                    f.create_dataset(name, data=arr)
                else:
                    raise NameError(
                        "A dataset with this name already exists, use force=True to override it"
                    )
            else:
                f.create_dataset(name, data=arr)

    def make(self, load_path: str, tmax: Optional[int] = None) -> None:
        """Automatically parse the load_path and will create datasets"""
        self._create_h5()
        out_path, mx3_path = self._get_paths(load_path)
        self.add_table(f"{out_path}/table.txt")
        self.add_mx3(mx3_path)
        dset_prefixes = self._get_dset_prefixes(out_path)
        for dset_prefix in dset_prefixes:
            self.add_dset(out_path, dset_prefix, tmax=tmax)

    @property
    def freqs(self) -> np.ndarray:
        """returns frequencies in GHz depending on the number of t points in the dset and value of dt"""
        with h5py.File(self.h5_path, "r") as f:
            freqs = np.fft.rfftfreq(f["disp"].shape[0], self.dt * 1e9)

        return freqs

    @property
    def kvecs(self) -> np.ndarray:
        """returns wavevectors"""
        with h5py.File(self.h5_path, "r") as f:
            kvecs = (
                np.fft.fftshift(np.fft.fftfreq(f["disp"].shape[1], self.dx) * 2 * np.pi)
                * 1e-6
            )
        return kvecs

    def start_dask_client(self, port=23232):
        ram = int(psutil.virtual_memory().free / 1e9 * 0.95)
        print(f"Dask client started at 127.0.0.1:{port} with {ram} GB of ram")
        client = Client(
            processes=False,
            threads_per_worker=mp.cpu_count(),
            n_workers=1,
            memory_limit=f"{ram} GB",
        )
        return client

    def disp(
        self,
        dset: str = "WG",
        name: str = "disp",
        slices: Tuple[Union[int, slice], ...] = (
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            2,
        ),
        save: bool = True,
        force: bool = False,
    ) -> None:
        """Calculates and returns the dispersions using dask"""
        if name in self.dsets():
            if force:
                with h5py.File(self.h5_path, "a") as f:
                    del f[name]
            else:
                input_string: str = input(
                    f"{name} is already a dataset, [y] to overwrite, [n] to cancel, else [input] a new name"
                )
                if input_string.lower() == "y":
                    with h5py.File(self.h5_path, "a") as f:
                        del f[name]
                elif input_string.lower() == "n":
                    return
                else:
                    name = input_string
        dask_client = self.start_dask_client()
        with h5py.File(self.h5_path, "r") as f:
            arr = da.from_array(f[dset], chunks=(None, None, 15, None, None))
            arr = arr[slices]  # slice
            arr = da.multiply(
                arr, np.hanning(arr.shape[0])[:, None, None, None]
            )  # hann filter on the t axis
            arr = arr.sum(axis=1)  # t,z,y,x => t,y,x sum of z
            arr = da.moveaxis(arr, 1, 0)  # t,y,x => y,t,x swap t and y
            ham2d = np.sqrt(
                np.outer(np.hanning(arr.shape[1]), np.hanning(arr.shape[2]))
            )  # shape(t, x)
            arr = da.multiply(arr, ham2d[None, :, :])  # hann window on t and x
            arr = da.fft.fft2(arr)  # 2d fft on t and x
            arr = da.subtract(
                arr, da.average(arr, axis=(1, 2))[:, None, None]
            )  # substract the avr of t,x for a given y
            arr = da.moveaxis(arr, 0, 1)
            arr = arr[: arr.shape[0] // 2]  # split f in 2, take 1st half
            arr = da.fft.fftshift(arr, axes=(1, 2))
            arr = da.absolute(arr)  # from complex to real
            arr = da.sum(arr, axis=1)  # sum y
            out = arr.compute()
        dask_client.close()

        if save:
            with h5py.File(self.h5_path, "a") as f:
                dset_disp = f.create_dataset(name, data=out)
                dset_disp.attrs["slices"] = str(slices)
                dset_disp.attrs["dset"] = dset

    def fft(
        self,
        dset: str = "ND",
        name: str = "fft",
        slices: Tuple[Union[int, slice], ...] = (
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            2,
        ),
        save: bool = True,
    ) -> np.ndarray:
        """Calculates and return the fft of the dataset"""
        with h5py.File(self.h5_path, "a") as f:
            if slices is None:
                arr = f[dset][:]
            else:
                arr = f[dset][slices]
            arr -= arr[0]

            for i in [0, 2, 3]:
                if arr.shape[i] % 2 == 0:
                    arr = np.delete(arr, 1, i)

            hann = np.hanning(arr.shape[0])
            for i, _ in enumerate(hann):
                arr[i] *= hann[i]

            arr = arr.sum(axis=1)  # t,z,y,x,c => t,y,x,c
            fft = []  # fft for each cell and comp
            for y in tqdm(
                range(arr.shape[1]),
                desc="Calculating FFT",
                total=arr.shape[1],
                leave=False,
            ):  # y
                for x in range(arr.shape[2]):  # x
                    for c in range(arr.shape[3]):
                        d = arr[:, y, x, c]
                        d = d - np.average(d)
                        fft.append(np.fft.rfft(d))
            out = np.array(fft)
            out = np.abs(out)
            out = np.sum(out, axis=0)
            out /= (
                arr.shape[1] * arr.shape[2]
            )  # changing the amplitude on a per cell basis

            if save:
                dset_fft = f.create_dataset(name, data=out)
                dset_fft.attrs["slices"] = str(slices)
                dset_fft.attrs["dset"] = dset
        return out

    def dplot(
        self,
        dset: str = "disp",
        fmin: int = 25,
        fmax: int = 450,
        kwidth: int = 150,
        dpi: int = 150,
    ):
        fig, ax = plt.subplots(1, 1, dpi=dpi)
        kmid = self.shape(dset)[1] // 2
        kmin = kmid - kwidth
        kmax = kmid + kwidth
        arr = self[dset][fmin:fmax, kmin:kmax]
        kvecs = self.kvecs[kmin:kmax]
        freqs = self.freqs[fmin:fmax]
        ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            cmap="cmo.amp",
            extent=[kvecs.min(), kvecs.max(), freqs.min(), freqs.max()],
        )
        ax.set_ylabel("Frequency (GHz)")
        ax.set_ylabel(r"Wavevectors (nm$^{-1}$)")
        fig.colorbar(ax.get_images()[0], ax=ax)
        return ax

    def save_ovf(self, dset: str, name: str, t: int = 0) -> None:
        """Saves the given dataset to a valid OOMMF V2 ovf file"""

        def whd(s):
            s += "\n"
            f.write(s.encode("ASCII"))

        arr = self[dset][t]
        out = arr.astype("<f4")
        out = out.tobytes()
        title = dset
        xstepsize, ystepsize, zstepsize = (
            self["dx"],
            self["dy"],
            self["dz"],
        )
        xnodes, ynodes, znodes = arr.shape[2], arr.shape[1], arr.shape[0]
        xmin, ymin, zmin = 0, 0, 0
        xmax, ymax, zmax = xnodes * xstepsize, ynodes * ystepsize, znodes * zstepsize
        xbase, ybase, _ = xstepsize / 2, ystepsize / 2, zstepsize / 2
        valuedim = arr.shape[-1]
        valuelabels = "x y z"
        valueunits = "1 1 1"
        total_sim_time = "0"
        with open(name, "wb") as f:
            whd("# OOMMF OVF 2.0")
            whd("# Segment count: 1")
            whd("# Begin: Segment")
            whd("# Begin: Header")
            whd(f"# Title: {title}")
            whd("# meshtype: rectangular")
            whd("# meshunit: m")
            whd(f"# xmin: {xmin}")
            whd(f"# ymin: {ymin}")
            whd(f"# zmin: {zmin}")
            whd(f"# xmax: {xmax}")
            whd(f"# ymax: {ymax}")
            whd(f"# zmax: {zmax}")
            whd(f"# valuedim: {valuedim}")
            whd(f"# valuelabels: {valuelabels}")
            whd(f"# valueunits: {valueunits}")
            whd(f"# Desc: Total simulation time:  {total_sim_time}  s")
            whd(f"# xbase: {xbase}")
            whd(f"# ybase: {ybase}")
            whd(f"# zbase: {ybase}")
            whd(f"# xnodes: {xnodes}")
            whd(f"# ynodes: {ynodes}")
            whd(f"# znodes: {znodes}")
            whd(f"# xstepsize: {xstepsize}")
            whd(f"# ystepsize: {ystepsize}")
            whd(f"# zstepsize: {zstepsize}")
            whd("# End: Header")
            whd("# Begin: Data Binary 4")
            f.write(struct.pack("<f", 1234567.0))
            f.write(out)
            whd("# End: Data Binary 4")
            whd("# End: Segment")
