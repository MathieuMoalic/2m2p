import numpy as np

import dask.array as da
from dask.diagnostics import ProgressBar

from ..base import Base


class modes(Base):
    def calc(self, dset: str = "m", name=None, slices=(slice(None))):
        if name is None:
            name = dset
        with ProgressBar():
            self.m.rm(f"modes/{name}")
            self.m.rm(f"fft/{name}")
            x1 = da.from_zarr(self.m[dset])
            x1 = x1[slices]
            if "stable" in self.m:
                x1 -= da.from_zarr(self.m.stable)[:]
            x1 = x1.rechunk((x1.shape[0], 1, 64, 64, x1.shape[-1]))
            x2 = da.fft.rfft(x1, axis=0)
            d1 = self.m.create_dataset(
                f"modes/{name}/arr",
                shape=x2.shape,
                chunks=(1, None, None, None, None),
                dtype=np.complex64,
            )
            da.to_zarr(x2, d1)
            x1 -= da.average(x1)
            x1 = x1 * np.hanning(x1.shape[0])[:, None, None, None, None]
            x1 = np.fft.rfft(x1, axis=0)
            x1 = da.absolute(x1)
            fft_max = da.max(x1, axis=(1, 2, 3))
            # fft_sum = da.sum(x1, axis=(1, 2, 3))
            da.to_zarr(
                fft_max,
                self.m.create_dataset(
                    f"fft/{name}/max",
                    shape=fft_max.shape,
                    chunks=None,
                    dtype=np.float32,
                ),
            )
            # da.to_zarr(
            #     fft_sum,
            #     self.m.create_dataset(
            #         f"fft/{name}/sum",
            #         shape=fft_sum.shape,
            #         chunks=None,
            #         dtype=np.float32,
            #     ),
            # )
        ts = self.m.m.attrs["t"][:]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9
        self.m.create_dataset(f"fft/{name}/freqs", data=freqs, chunks=False)
        self.m.create_dataset(f"modes/{name}/freqs", data=freqs, chunks=False)

    def calc2(self, dset: str, force=False, name=None, tmax=None):
        if name is None:
            name = dset
        self.m.check_path(f"modes/{name}/freqs", force)
        self.m.check_path(f"modes/{name}/arr", force)
        ts = self.m[dset].attrs["t"][:]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9
        self.m.create_dataset(
            f"modes/{name}/freqs", data=freqs, chunks=False, compressor=False
        )
        with ProgressBar():
            arr = da.from_array(self.m[dset], chunks=(None, None, 16, None, None))
            arr = arr[:tmax]
            arr = da.fft.rfft(arr, axis=0)  # pylint: disable=unexpected-keyword-arg
            d = self.m.create_dataset(
                f"modes/{name}/arr",
                shape=arr.shape,
                chunks=(1, None, None, None, None),
                dtype=np.complex128,
            )
            da.to_zarr(arr, d)
