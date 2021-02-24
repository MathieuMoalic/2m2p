import h5py
import numpy as np

from m2p2._make import Make
from m2p2._transform import Transform


class H5(Make, Transform):
    def __init__(self, path: str, loadpath: str = None):
        self.path = path
        self.name = path.split("/")[-1]
        if loadpath is not None:
            self.make(loadpath)

    def __getitem__(self, index):
        with h5py.File(self.path, "r") as f:
            if type(index) == str:
                if index in list(f.keys()):  # if dset
                    self._dset = index
                    return self
                elif index in list(f.attrs.keys()):  # if attrs
                    return f.attrs[index]
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
