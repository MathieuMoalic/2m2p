import numpy as np

import dask.array as da
from dask.diagnostics import ProgressBar

from ..base import Base


class modes(Base):
    def calc(self, dset: str, force=False, name=None, tmax=None):
        if name is None:
            name = dset
        self.llyr.check_path(f"modes/{name}/freqs", force)
        self.llyr.check_path(f"modes/{name}/arr", force)
        ts = self.llyr[dset].attrs["t"][:]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts)) * 1e-9
        self.llyr.create_dataset(
            f"modes/{name}/freqs", data=freqs, chunks=False, compressor=False
        )
        with ProgressBar():
            arr = da.from_array(self.llyr[dset], chunks=(None, None, 16, None, None))
            arr = arr[:tmax]
            arr = da.fft.rfft(arr, axis=0)  # pylint: disable=unexpected-keyword-arg
            d = self.llyr.create_dataset( f"modes/{name}/arr",shape=arr.shape, chunks=(1,None,None,None,None),dtype=np.complex128)
            da.to_zarr(arr,d)
