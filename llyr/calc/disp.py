from typing import Optional

import numpy as np

from ..base import Base


class disp(Base):
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
    ):
        if name is None:
            name = dset_name
        if force:
            self.llyr.rm(f"disp/{name}")
        if any(
            f"disp/{name}/{dset_name}" in self.llyr
            for dset in ["freqs", "kvecs", "disp", "fft2d"]
        ):
            raise NameError(
                f"The dataset:'disp/{name}' already exists, you can use 'force=True'"
            )
        dset = self.llyr[dset_name]
        if tslice.stop is None or tslice.stop > dset.shape[0]:
            tslice = slice(dset.shape[0])

        arr = dset[tslice, zslice, yslice, xslice, cslice]
        if arr.shape[3] % 2 == 0:
            arr = arr[:, :, :, 1:, :]
        if arr.shape[0] % 2 == 0:
            arr = arr[1:]
        arr *= np.hanning(arr.shape[0])[:, None, None, None, None]
        arr -= arr[0]
        arr = np.sum(arr, axis=1)
        # hann window on t and x => t,y,x,c
        hann2d = np.outer(np.hanning(arr.shape[0]), np.hanning(arr.shape[2]))
        arr *= np.sqrt(hann2d)[:, None, :, None]
        # 2d fft on t and x => f,y,kx,c
        arr = np.fft.fft2(arr, axes=[0, 2])
        self.llyr.create_dataset(f"disp/{name}/fft2d", data=arr, chunks=None)
        # substract the avr of t,x for a given y  => f,y,kx,c
        arr -= np.average(arr, axis=(0, 2))[None, :, None, :]
        # split f in 2, take 1st half => f,y,kx,c
        arr = arr[: arr.shape[0] // 2]
        arr = np.fft.fftshift(arr, axes=(1, 2))
        arr = np.abs(arr)  # from complex to real
        arr = np.sum(arr, axis=1)  # sum y => f,kx,c
        self.llyr.create_dataset(f"disp/{name}/disp", data=arr, chunks=None)

        ts = dset.attrs["t"][tslice]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts))
        self.llyr.create_dataset(f"disp/{name}/freqs", data=freqs, chunks=None)

        kvecs = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], self.llyr.dx)) * 2 * np.pi
        self.llyr.create_dataset(f"disp/{name}/kvecs", data=kvecs, chunks=None)
