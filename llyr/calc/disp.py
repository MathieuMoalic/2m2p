from typing import Optional

import numpy as np
import dask.array as da
import h5py

from ..base import Base


class disp(Base):
    def calc(
        self,
        dset: str,
        name: Optional[str] = None,
        force: Optional[bool] = False,
        tslice=slice(None),
        zslice=slice(None),
        yslice=slice(None),
        xslice=slice(None),
        cslice=2,
    ):
        if name is None:
            name = dset
        self.llyr.check_path(f"modes/{name}/arr", force)
        self.llyr.check_path(f"modes/{name}/freqs", force)
        self.llyr.check_path(f"modes/{name}/kvecs", force)
        with h5py.File(self.llyr.apath, "a") as f:
            arr = da.from_array(f[dset], chunks=(None, None, 16, None, None))
            arr = arr[(tslice, zslice, yslice, xslice, cslice)]  # slice
            s = arr.shape
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
            arr.to_hdf5(self.llyr.apath, f"disp/{name}/arr")

        freqs = np.fft.rfftfreq(s[0], self.llyr.dt)
        kvecs = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], self.llyr.dx)) * 1e-6
        self.llyr.z.create_dataset(
            f"modes/{name}/arr", data=arr, chunks=(1, None, None, None, None)
        )
        self.llyr.z.create_dataset(f"modes/{name}/freqs", data=freqs, chunks=(None))
        self.llyr.z.create_dataset(f"modes/{name}/kvecs", data=kvecs, chunks=(None))

        return arr
