from typing import Optional

import numpy as np

from ..base import Base


class fft(Base):
    def calc(
        self,
        dset_name: str,
        name: Optional[str] = None,
        force: Optional[bool] = False,
        tslice=slice(None),
        zslice=slice(None),
        yslice=slice(None),
        xslice=slice(None),
        cslice=slice(None),
        zero=None,
    ):
        if name is None:
            name = dset_name
        if force:
            self.m.rm(f"fft/{name}")
        if any(f"fft/{name}/{d}" in self.m for d in ["freqs", "fft"]):
            raise NameError(
                f"The dataset:'fft/{name}' already exists, you can use 'force=True'"
            )
        dset = self.m[dset_name]
        if tslice.stop is None or tslice.stop > dset.shape[0]:
            tslice = slice(dset.shape[0])
        arr = dset[(tslice, zslice, yslice, xslice, cslice)]
        if zero is None:
            arr -= arr[0]
        else:
            arr -= zero
        arr -= np.average(arr)
        arr *= np.hanning(arr.shape[0])[:, None, None, None, None]
        arr = np.fft.rfft(arr, axis=0)
        arr = np.abs(arr)
        arr = np.max(arr, axis=(1, 2, 3))
        self.m.create_dataset(
            f"fft/{name}/fft", data=arr, chunks=False, compressor=False
        )

        ts = dset.attrs["t"][tslice]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts))
        self.m.create_dataset(
            f"fft/{name}/freqs", data=freqs, chunks=False, compressor=False
        )
