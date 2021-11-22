from typing import Optional

import numpy as np
import dask.array as da
import h5py

from ..base import Base


class fft(Base):
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
        self.llyr.check_path(f"fft/{name}/arr", force)
        self.llyr.check_path(f"fft/{name}/freqs", force)
        with h5py.File(self.llyr.apath, "r") as f:
            arr = da.from_array(f[dset], chunks=(None, None, 16, None, None))
            arr = arr[(tslice, zslice, yslice, xslice, cslice)]
            s = arr.shape
            arr = arr.sum(axis=1)  # sum all z
            arr = da.subtract(arr, arr[0])
            arr = da.subtract(arr, da.average(arr, axis=0)[None, :])
            arr = da.multiply(arr, np.hanning(arr.shape[0])[:, None, None])
            arr = da.swapaxes(arr, 0, -1)
            arr = da.reshape(
                arr, (arr.shape[0] * arr.shape[1], arr.shape[2])
            )  # flatten all the cells
            arr = da.fft.rfft(arr)
            arr = da.absolute(arr)
            arr = da.sum(arr, axis=0)
            # da.to_zarr(arr, f"fft/{name}/arr")
            arr.to_zarr(self.llyr.apath, f"fft/{name}/arr")

        freqs = np.fft.rfftfreq(s[0], self.llyr.dt)
        self.llyr.create_dataset(
            f"fft/{name}/freqs", data=freqs, chunks=False, compressor=False
        )
