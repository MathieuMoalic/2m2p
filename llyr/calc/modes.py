import numpy as np

import dask.array as da
import h5py

from ..base import Base


class modes(Base):
    def calc(self, dset: str, override=False, name=None, tmax=None):
        if name is None:
            name = dset
        self.llyr.check_path(f"modes/{name}/arr", override)
        self.llyr.check_path(f"modes/{name}/freqs", override)
        with h5py.File(self.llyr.path, "a") as f:
            arr = da.from_array(f[dset], chunks=(None, None, 16, None, None))
            arr = arr[:tmax]
            arr = da.fft.rfft(arr, axis=0)  # pylint: disable=unexpected-keyword-arg
            arr = arr.compute()
            arr.to_hdf5(self.llyr.path, f"modes/{name}/arr")
        freqs = np.fft.rfftfreq(arr.shape[0], self.llyr.dt) * 1e-9
        self.llyr.h5.add_dset(freqs, f"modes/{name}/freqs", override=override)
