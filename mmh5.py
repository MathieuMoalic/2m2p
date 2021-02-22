import h5py
import numpy as np
from glob import glob
from mmpp.ovf import ovfParms
from mmpp.marray import Marray
import os
import re
import gc
from tqdm.notebook import tqdm
import multiprocessing as mp
import dask.array as da


class h5:
    def __init__(self, path: str, loadpath: str = None):
        self.path = path
        if loadpath is not None:
            self.make(loadpath)

    def __getitem__(self, index):
        with h5py.File(self.path, "r") as f:
            if type(index) == str:
                if index in list(f.keys()):  # if dset
                    self._dset = index
                    return self
                elif index in list(f.attrs.keys()):  # if attrs
                    print(f.attrs[index])
                    # return f.attrs[index]
                else:
                    print("Not Found")
            else:
                return f[self._dset][index]

    def dset(self, name: str, data):
        with h5py.File(self.path, "a") as f:
            f.create_dataset(name, data=data)

    def a(self, name: str, key: str, val):
        with h5py.File(self.path, "a") as f:
            f[name].attrs[key] = val

    @property
    def p(self):
        with h5py.File(self.path, "r") as f:
            print("Datasets:")
            for key in list(f.keys()):
                print(f"    {key:<15}: {f[key].shape}")
            print("Attributes:")
            for key in list(f.attrs.keys()):
                if key != "script":
                    print(f"    {key:<15}= {f.attrs[key]}")

    def script(self, savepath=None):
        with h5py.File(self.path, "r") as f:
            if savepath is None:
                print("dwad")
                print(self["script"])

    def freqs(self):
        with h5py.File(self.path, "r") as f:
            dt = f.attrs["dt"]
            tshape = f["WG"].shape[0]
        return np.fft.rfftfreq(tshape, dt * 1e9)

    def kvecs(self):
        with h5py.File(self.path, "r") as f:
            dx = f.attrs["dx"]
            xshape = f["WG"].shape[3]
        return np.fft.fftshift(np.fft.fftfreq(xshape, dx) * 2 * np.pi) / 1e6

    def d(self, dset: str):
        with h5py.File(self.path, "a") as f:
            del f[dset]

    def move(self, source: str, destination: str):
        with h5py.File(self.path, "a") as f:
            f.move(source, destination)

    def load_ovf(self, path: str):
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

    def make(self, loadpath: str):
        if os.path.isfile(self.path):
            c = input(f"{self.path} already exists, overwrite it [y/n]?")
            if c.lower() in ["y", "yes"]:
                with h5py.File(self.path, "w") as f:
                    pass
            else:
                return

        with h5py.File(self.path, "a") as f:
            # save script
            with open(f"{loadpath}.mx3", "r") as script:
                f.attrs["script"] = script.read()

            # save table.txt
            if os.path.isfile(f"{loadpath}.out/table.txt"):
                with open(f"{loadpath}.out/table.txt", "r") as table:
                    header = table.readline()
                    tab = np.loadtxt(table).T
                    tableds = f.create_dataset("table", data=tab)
                    tableds.attrs["header"] = header
                    dt = (tab[0, -1] - tab[0, 0]) / (tab.shape[1] - 1)
                    f.attrs["dt"] = dt

            # get names of datasets
            heads = set(
                [os.path.basename(x)[:-8] for x in glob(f"{loadpath}.out/*.ovf")]
            )
            dset_names = []
            for i, head in enumerate(heads):
                if "zrange4" in head:
                    dset_names.append("ND")
                elif "zrange2" in head:
                    dset_names.append("WG")
                elif "st" in head:
                    dset_names.append("stable")
                else:
                    input_name = input(f"{head} :")
                    if input_name == "del":
                        del heads[i]
                    else:
                        dset_names.append(input_name)

            # get list of ovf files and shapes
            ovf_list = []
            ovf_shape = []
            for head in heads:
                L = glob(f"{loadpath}.out/{head}*.ovf")
                ovf_list.append(L)
                shape = self.load_ovf(L[0]).shape
                ovf_shape.append((len(L),) + shape)

            # save datasets
            for ovfs, shape, dset_name in zip(ovf_list, ovf_shape, dset_names):
                dset = f.create_dataset(dset_name, shape, np.float32)
                pool = mp.Pool(processes=int(mp.cpu_count() - 1))
                for i, d in enumerate(pool.imap(loadovf, ovfs)):
                    dset[i] = d
                pool.close()
                pool.join()

            # get attrs
            for line in f.attrs["script"].split("\n"):
                line = line.lower()
                if re.search(r"[d][xyz] +:=", line) is not None:
                    key = re.search(r"[d][xyz]", line).group()
                    value = float(re.search(r"[\d.e-]+", line).group())
                    f.attrs[key] = value

                if "setregion" in line.lower() and "anisu" not in line:
                    try:
                        key = (
                            re.search(r"\w+_region", line).group().split("_")[0]
                            + "_"
                            + re.search(r"\w+", line).group()
                        )
                        value = float(re.search(r"\d+\.?\d*e?-?\d*", line[4:]).group())
                        f.attrs[key] = value
                    except:
                        pass

                if re.search(r"amps +:=", line) is not None:
                    f.attrs["amps"] = float(re.search(r"[\d.e-]+", line).group())

                if re.search(r"f +:=", line) is not None:
                    f.attrs["f"] = float(re.search(r"[\d.e-]+", line).group())

    def disp(
        self,
        dset: str = "WG",
        name: str = "disp",
        slices: tuple = (slice(None), slice(None), slice(None), slice(None), 2),
        save: bool = True,
    ):
        with h5py.File(f"{sims}/sinc/sd_wg.h5", "r") as f:
            arr = da.from_array(f["WG"], chunks=(None, None, 15, None, None))
            arr = arr[:, :, :, :, 1]  # slice
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
            arr = arr.compute()
            if save:
                dset_disp = f.create_dataset(name, data=disp)
                dset_disp.attrs["slices"] = str(slices)
                dset_disp.attrs["dset"] = dset

        return disp

    def fft(
        self,
        dset: str = "ND",
        name: str = "fft",
        slices: tuple = (slice(None), slice(None), slice(None), slice(None), 2),
        save: bool = True,
    ):
        with h5py.File(self.path, "a") as f:
            if slices is None:
                arr = f[dset][:]
            else:
                arr = f[dset][slices]
            arr -= arr[0]

            for i in [0, 2, 3]:
                if arr.shape[i] % 2 == 0:
                    arr = np.delete(arr, 1, i)

            hann = np.hanning(arr.shape[0])
            for i in range(len(hann)):
                arr[i] *= hann[i]

            _hy = np.hamming(arr.shape[2])
            _hx = np.hamming(arr.shape[3])
            a = np.sqrt(np.outer(_hy, _hx))
            pre_shape = arr.shape
            mxy = (
                np.reshape(
                    arr,
                    [
                        arr.shape[0] * arr.shape[1] * arr.shape[-1],
                        arr.shape[2],
                        arr.shape[3],
                    ],
                )
                * a
            )
            arr = np.reshape(arr, pre_shape)

            arr = arr.sum(axis=1)  # t,z,y,x,c => t,y,x,c
            fft = []  # fft fot each cell and comp
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
            fft = np.abs(fft)
            fft = np.sum(fft, axis=0)
            fft /= (
                arr.shape[1] * arr.shape[2]
            )  # changing the amplitude on a per cell basis

            if save:
                dset_fft = f.create_dataset(name, data=fft)
                dset_fft.attrs["slices"] = str(slices)
                dset_fft.attrs["dset"] = dset
        del arr
        gc.collect()
        return fft
