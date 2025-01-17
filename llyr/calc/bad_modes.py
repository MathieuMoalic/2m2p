import numpy as np
import dask.array as da

from ..base import Base


class bad_modes(Base):
    def calc(self, dset: str = "m", name=None, slices=(slice(None),)):
        if name is None:
            name = dset
        self.m.rm(f"fft_bad/{name}/sum")
        x1 = da.from_zarr(self.m[dset])
        if slices[0] == slice(None):
            slices = list(slices)
            slices[0] = slice(None, self.m[dset].shape[0])
            slices = tuple(slices)
        x1 = x1[slices]
        if "stable" in self.m:
            x1 -= da.from_zarr(self.m.stable)[:1]
        x1 = x1.rechunk((x1.shape[0], 1, 64, 64, x1.shape[-1]))
        x1 -= da.average(x1)
        x1 = x1 * np.hanning(x1.shape[0])[:, None, None, None, None]
        x1 = np.fft.rfft(x1, axis=0)
        x1 = da.absolute(x1)
        fft_max = da.sum(x1, axis=(1, 2, 3))
        da.to_zarr(
            fft_max,
            self.m.create_dataset(
                f"fft/{name}/bad",
                shape=fft_max.shape,
                chunks=None,
                dtype=np.float32,
            ),
        )
        ts = self.m.m.attrs["t"][slices[0]]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9
        self.m.create_dataset(f"fft/{name}/freqs", data=freqs, chunks=False)
        self.m.create_dataset(f"modes/{name}/freqs", data=freqs, chunks=False)
