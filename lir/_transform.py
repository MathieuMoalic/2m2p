import multiprocessing as mp
import dask.array as da 
import numpy as np
import h5py
from typing import Tuple, Union


class Transform:

    def disp(
        self,
        dset: str = "WG",
        name: str = "disp",
        slices: Tuple[Union[int, slice], ...] = (
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            2,
        ),
        save: bool = True,
    ) -> np.ndarray:
        """Calculates and returns the dispersions using dask"""
        with h5py.File(self.h5_path, "r") as f:
            arr = da.from_array(f[dset], chunks=(None, None, 15, None, None))
            arr = arr[slices]  # slice
            arr = da.multiply(
                arr, np.hanning(arr.shape[0])[:, None, None, None]
            )  # hann filter on the t axis
            arr = arr.sum(axis=1)  # t,z,y,x => t,y,x sum of z
            arr = da.moveaxis(arr, 1, 0)  # t,y,x => y,t,x swap t and y
            ham2d = np.sqrt(
                np.outer(np.hanning(arr.shape[1]), np.hanning(arr.shape[2]))
            )  # shape(t, x)
            arr = da.multiply(arr, ham2d[None, :, :])  # hann window on t and x
            arr = da.fft.fft2(arr)  # 2d fft on t and x
            arr = da.subtract(
                arr, da.average(arr, axis=(1, 2))[:, None, None]
            )  # substract the avr of t,x for a given y
            arr = da.moveaxis(arr, 0, 1)
            arr = arr[: arr.shape[0] // 2]  # split f in 2, take 1st half
            arr = da.fft.fftshift(arr, axes=(1, 2))
            arr = da.absolute(arr)  # from complex to real
            arr = da.sum(arr, axis=1)  # sum y
            out = arr.compute()

        if save:
            with h5py.File(self.h5_path, "a") as f:
                dset_disp = f.create_dataset(name, data=out)
                dset_disp.attrs["slices"] = str(slices)
                dset_disp.attrs["dset"] = dset

        return out

    def fft(
        self,
        dset: str = "ND",
        name: str = "fft",
        slices: Tuple[Union[int, slice], ...] = (
            slice(None),
            slice(None),
            slice(None),
            slice(None),
            2,
        ),
        save: bool = True,
    ) -> np.ndarray:
        """Calculates and return the fft of the dataset"""
        with h5py.File(self.h5_path, "a") as f:
            if slices is None:
                arr = f[dset][:]
            else:
                arr = f[dset][slices]
            arr -= arr[0]

            for i in [0, 2, 3]:
                if arr.shape[i] % 2 == 0:
                    arr = np.delete(arr, 1, i)

            hann= np.hanning(arr.shape[0])
            for i in range(len(hann)):
                arr[i] *= hann[i]

            _hy = np.hamming(arr.shape[2])
            _hx = np.hamming(arr.shape[3])
            a = np.sqrt(np.outer(_hy, _hx))
            pre_shape: Tuple[Union[int, slice], ...] = arr.shape
            mxy = (
                np.reshape(
                    arr,
                    [
                        arr.shape[0] * arr.shape[1] * arr.shape[-1],
                        arr.shape[2],
                        arr.shape[3],
                    ],
                )
                * a
            )
            arr = np.reshape(arr, pre_shape)

            arr = arr.sum(axis=1) # t,z,y,x,c => t,y,x,c
            fft = []  # fft for each cell and comp
            for y in tqdm(
                range(arr.shape[1]),
                desc="Calculating FFT",
                total=arr.shape[1],
                leave=False,
            ):  # y
                for x in range(arr.shape[2]):  # x
                    for c in range(arr.shape[3]):
                        d = arr[:, y, x, c]
                        d = d - np.average(d)
                        fft.append(np.fft.rfft(d))
            out = np.array(fft)
            out = np.abs(out)
            out = np.sum(out, axis=0)
            out /= (
                arr.shape[1] * arr.shape[2]
            )  # changing the amplitude on a per cell basis

            if save:
                dset_fft = f.create_dataset(name, data=out)
                dset_fft.attrs["slices"] = str(slices)
                dset_fft.attrs["dset"] = dset
        return out