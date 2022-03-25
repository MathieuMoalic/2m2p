from typing import Optional

import numpy as np

from ..base import Base


class disp(Base):
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
        self.llyr.check_path(f"disp/{name}/arr", force)
        self.llyr.check_path(f"disp/{name}/freqs", force)
        self.llyr.check_path(f"disp/{name}/kvecs", force)
        arr = self.llyr[dset][tslice, zslice, yslice, xslice, cslice]
        ts = self.llyr[dset].attrs["t"][tslice]
        freqs = np.fft.rfftfreq(len(ts), (ts[-1] - ts[0]) / len(ts))
        kvecs = np.fft.fftshift(np.fft.fftfreq(arr.shape[3], self.llyr.dx)) * 2 * np.pi
        arr *= np.hanning(arr.shape[0])[:, None, None, None, None]
        arr -= arr[0]
        arr = np.sum(arr, axis=1)  # sum z
        arr = np.moveaxis(arr, 1, 0)  # t,y,x => y,t,x swap t and y
        arr *= np.sqrt(np.outer(np.hanning(arr.shape[1]), np.hanning(arr.shape[2])))[
            None, :, :, None
        ]  # hann window on t and x
        arr = np.fft.fft2(arr, axes=[1, 2])  # 2d fft on t and x
        arr -= np.average(arr, axis=(1, 2))[
            :, None, None
        ]  # substract the avr of t,x for a given y
        arr = np.moveaxis(arr, 0, 1)
        arr = arr[: arr.shape[0] // 2]  # split f in 2, take 1st half
        arr = np.fft.fftshift(arr, axes=(1, 2))
        arr = np.abs(arr)  # from complex to real
        arr = np.sum(arr, axis=1)  # sum y

        self.llyr.create_dataset(f"disp/{name}/arr", data=arr, chunks=None)
        self.llyr.create_dataset(f"disp/{name}/freqs", data=freqs, chunks=None)
        self.llyr.create_dataset(f"disp/{name}/kvecs", data=kvecs, chunks=None)
