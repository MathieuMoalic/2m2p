import numpy as np

import dask.array as da
from dask.diagnostics import ProgressBar
import h5py

from ..base import Base


class modes(Base):
    def calc(self, dset: str, force=False, name=None, tmax=None):
        if name is None:
            name = dset
        self.llyr.check_path(f"modes/{name}/arr", force)
        self.llyr.check_path(f"modes/{name}/freqs", force)
        with h5py.File(self.llyr.path, "a") as f:
            with ProgressBar():
                arr = da.from_array(f[dset], chunks=(None, None, 16, None, None))
                arr = arr[:tmax]
                s = arr.shape
                arr = da.fft.rfft(arr, axis=0)  # pylint: disable=unexpected-keyword-arg
                arr.to_hdf5(self.llyr.path, f"modes/{name}/arr")
        freqs = np.fft.rfftfreq(s[0], self.llyr.dt) * 1e-9
        self.llyr.h5.add_dset(freqs, f"modes/{name}/freqs", force=force)
