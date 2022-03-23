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
        cslice=slice(None),
    ):
        if name is None:
            name = dset
        self.llyr.check_path(f"fft/{name}/arr", force)
        self.llyr.check_path(f"fft/{name}/freqs", force)
        ts = self[dset].attrs["t"][:]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts))
        arr = self.llyr[dset][(tslice, zslice, yslice, xslice, cslice)]
        arr -= arr[0]
        arr -= np.average(arr)
        arr *= np.hanning(arr.shape[0])[:,None,None,None,None]
        arr = np.fft.rfft(arr,axis=0)
        arr = np.abs(arr)
        arr = np.sum(arr,axis=(1,2,3))
        m.create_dataset(f"fft/{name}/arr",shape=y.shape,chunks=None,dtype=np.int32,data=y)
        m.create_dataset(f"fft/{name}/freqs",shape=freqs.shape,chunks=None,dtype=np.int32,data=freqs)