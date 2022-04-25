import numpy as np

import dask.array as da
from dask.diagnostics import ProgressBar

from ..base import Base


class modes(Base):
    def calc(self, dset: str = "m", force=False, name=None, tmax=None):
        if name is None:
            name = dset
        with ProgressBar():
            self.llyr[dset].rm(f"modes/{dset}")
            self.llyr[dset].rm(f"fft/{dset}")
            x1 = da.from_zarr(self.llyr[dset])
            x1 = x1.rechunk((x1.shape[0], 1, 64, 64, x1.shape[-1]))
            d2 = self.llyr.create_dataset(
                f"modes/{dset}/arr",
                shape=x1.shape,
                chunks=(1, x1.shape[1], x1.shape[2], x1.shape[3], x1.shape[4]),
                dtype=np.complex128,
            )
            da.to_zarr(da.fft.rfft(x1, axis=0), d2)
            x1 -= da.average(x1)
            x1 = x1 * np.hanning(x1.shape[0])[:, None, None, None, None]
            x1 = np.fft.rfft(x1, axis=0)
            x1 = da.absolute(x1)
            x1 = da.max(x1, axis=(1, 2, 3))
            d2 = self.llyr.create_dataset(
                f"fft/{dset}/max", shape=x1.shape, chunks=None, dtype=np.float32
            )
            da.to_zarr(x1, d2)
        ts = self.llyr.m.attrs["t"][:]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts))
        self.llyr.create_dataset(f"fft/{dset}/freqs", data=freqs, chunks=False)
        self.llyr.create_dataset(f"modes/{dset}/freqs", data=freqs, chunks=False)
