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
        override: Optional[bool] = False,
        tslice=slice(None),
        zslice=slice(None),
        yslice=slice(None),
        xslice=slice(None),
        cslice=2,
    ):
        with h5py.File(self.llyr.path, "r") as f:
            arr = da.from_array(f[dset], chunks=(None, None, 16, None, None))
            arr = arr[(tslice, zslice, yslice, xslice, cslice)]  # slice
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

        freqs = np.fft.rfftfreq(arr.shape[0], self.llyr.dt)
        kvecs = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], self.llyr.dx)) * 1e-6
        if name is not None:
            self.llyr.add_dset(arr, f"{name}/arr", override)
            self.llyr.add_dset(freqs, f"{name}/freqs", override)
            self.llyr.add_dset(kvecs, f"{name}/kvecs", override)

        return arr
