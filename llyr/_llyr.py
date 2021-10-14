from typing import Optional, Union, Tuple
import os
import multiprocessing as mp
import colorsys

import numpy as np
import dask.array as da
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import cmocean  # pylint: disable=unused-import
import peakutils

from ._make import Make
from ._ovf import save_ovf, load_ovf
from ._utils import hsl2rgb


class Llyr:
    def __init__(self, path: str) -> None:
        path = os.path.abspath(path)
        if path[-4:] == ".out":
            self.path = path[:-4]
        else:
            self.path = path
        if self.path[-3:] != ".h5":
            self.path = f"{self.path}.h5"
        else:
            self.path = path
        self.name = self.path.split("/")[-1].replace(".h5", "")
        self._getitem_dset: Optional[str] = None

    def make(
        self,
        load_path: Optional[str] = None,
        tmax: Optional[int] = None,
        override: bool = False,
        delete_out: bool = False,
        delete_mx3: bool = False,
        skip_ovf: bool = False,
    ):

        Make(self, load_path, tmax, override, delete_out, delete_mx3, skip_ovf)
        return self

    def __repr__(self) -> str:
        return f"Llyr('{self.name}')"

    def __str__(self) -> str:
        return f"Llyr('{self.name}')"

    def __getitem__(
        self,
        index: Union[str, Tuple[Union[int, slice], ...]],
    ) -> Union["Llyr", float, np.ndarray]:
        if isinstance(index, (slice, tuple, int)):
            # if dset is defined
            if isinstance(self._getitem_dset, str):
                out_dset: np.ndarray = self.get_dset(self._getitem_dset, index)
                self._getitem_dset = None
                return out_dset
            else:
                raise AttributeError("You can only slice datasets")

        elif isinstance(index, str):
            # if dataset
            if index in self.dsets:
                self._getitem_dset = index
                return self
            # if attribute
            elif index in self.attrs:
                out_attribute: float = self.attrs[index]
                return out_attribute
            else:
                raise KeyError("No such Dataset or Attribute")
        else:
            raise TypeError()

    @property
    def mx3(self) -> str:
        print(self["mx3"])

    @property
    def dt(self) -> float:
        return self.attrs["dt"]

    @property
    def dx(self) -> float:
        return self.attrs["dx"]

    @property
    def dy(self) -> float:
        return self.attrs["dy"]

    @property
    def dz(self) -> float:
        return self.attrs["dz"]

    @property
    def pp(self) -> None:
        print("Datasets:")
        for dset_name, dset_shape in self.dsets.items():
            print(f"    {dset_name:<20}: {dset_shape}")
        print("Global Attributes:")
        for key, val in self.attrs.items():
            if key in ["mx3", "script", "logs"]:
                val = val.replace("\n", "")
                print(f"    {key:<20}= {val[:10]}...")
            else:
                print(f"    {key:<20}= {val}")

    @property
    def p(self):
        dsets, groups = {}, {}
        for name, shape in self.dsets.items():
            if "/" in name:
                group, dset = name.split("/")
                if group not in groups:
                    groups[group] = {}
                groups[group][dset] = shape
            else:
                dsets[name] = shape
        a = ["" for i in range(1 + max(len(self.attrs), len(self.dsets)))]
        i = 0
        a[0] = "Datasets "
        d1 = len(a[0])
        for k, v in dsets.items():
            if i == 0:
                q = "┬"
            else:
                q = "├"
            padding1 = (d1 - len(a[i])) * " "
            a[i] += f"{padding1}{q}─ {k} : {v}"
            i += 1
        for k, v in groups.items():
            j = 0
            padding1 = (d1 - len(a[i])) * " "
            a[i] += f"{padding1}├─ {k} ┬"
            d2 = (len(a[i]) - d1 - 2) * " "
            length2 = len(v)
            for k2, v2 in v.items():
                if j == 0:
                    q2 = ""
                elif j == length2 - 1:
                    q2 = padding1 + " " + d2 + "└"
                else:
                    q2 = padding1 + " " + d2 + "├"
                a[i] += f"{q2}─ {k2} : {v2}"
                i += 1
                j += 1

        attrs = ["Global Attributes"]
        for key, val in self.attrs.items():
            if key in ["mx3", "script", "logs"]:
                val = val.replace("\n", "")
                attrs.append(f"    {key:<10}= {val[:10]}...")
            else:
                attrs.append(f"    {key:<10}= {val:.4}")
        for i in range(max(len(a), len(attrs))):
            if i >= len(attrs):
                attrs.append(45 * " ")
            elif i >= len(a):
                a.append(" ")
            else:
                diff_a = 45 - len(attrs[i])
                attrs[i] += diff_a * " "

        for i, _ in enumerate(a):
            a[i] = attrs[i] + a[i]
        for aa in a:
            print(aa)

    @property
    def dsets(self) -> dict:
        def add_dset(name, obj):
            # pylint: disable=protected-access
            if isinstance(obj, h5py._hl.dataset.Dataset):
                dsets[name] = obj.shape

        dsets = {}
        with h5py.File(self.path, "r") as f:
            f.visititems(add_dset)
        return dsets

    @property
    def attrs(self) -> dict:
        attrs = {}
        with h5py.File(self.path, "r") as f:
            for k, v in f.attrs.items():
                attrs[k] = v
        return attrs

    @property
    def stable(self) -> None:
        save_ovf(
            self.path.replace(".h5", ".ovf"),
            self["stable"][:],
            self.dx,
            self.dy,
            self.dz,
        )

    @property
    def snap(self):
        self.snapshot_png("stable")

    # h5 functions

    def save_as_ovf(self, arr: np.ndarray, name: str):
        path = self.path.replace(f"{self.name}", f"{name}.ovf")
        save_ovf(path, arr, self.dx, self.dy, self.dz)

    def create_h5(self, override: bool) -> bool:
        """Creates an empty .h5 file"""
        if override:
            with h5py.File(self.path, "w"):
                return True
        else:
            if os.path.isfile(self.path):
                input_string: str = input(
                    f"{self.path} already exists, overwrite it [y/n]?"
                )
                if input_string.lower() in ["y", "yes"]:
                    with h5py.File(self.path, "w"):
                        return True
        return False

    def shape(self, dset: str) -> Tuple:
        with h5py.File(self.path, "r") as f:
            return f[dset].shape

    def delete(self, dset: str) -> None:
        """deletes dataset"""
        with h5py.File(self.path, "a") as f:
            del f[dset]

    def move(self, source: str, destination: str) -> None:
        """move dataset or attribute"""
        with h5py.File(self.path, "a") as f:
            f.move(source, destination)

    def add_attr(
        self,
        key: str,
        val: Union[str, int, float, slice, Tuple[Union[int, slice], ...]],
        dset: Optional[str] = None,
    ) -> None:
        """set a new attribute"""
        if dset is None:
            with h5py.File(self.path, "a") as f:
                f.attrs[key] = val
        else:
            with h5py.File(self.path, "a") as f:
                f[dset].attrs[key] = val

    def add_dset(self, arr: np.ndarray, name: str, override: bool = False):
        if name in self.dsets:
            if override:
                self.delete(name)
            else:
                raise NameError(
                    f"Dataset with name '{name}' already exists, you can use 'override=True'"
                )
        with h5py.File(self.path, "a") as f:
            f.create_dataset(name, data=arr)

    def get_dset(self, dset, slices):
        with h5py.File(self.path, "r") as f:
            return f[dset][slices]

    def get_attrs(self, dset):
        with h5py.File(self.path, "r") as f:
            return dict(f[dset].attrs)

    def load_dset(self, name: str, dset_shape: tuple, ovf_paths: list) -> None:
        with h5py.File(self.path, "a") as f:
            dset = f.create_dataset(name, dset_shape, np.float32)
            with mp.Pool(processes=int(mp.cpu_count())) as p:
                for i, data in enumerate(
                    tqdm(
                        p.imap(load_ovf, ovf_paths),
                        leave=False,
                        desc=name,
                        total=len(ovf_paths),
                    )
                ):
                    dset[i] = data

    # post processing

    def disp(
        self,
        dset: str,
        name: Optional[str] = None,
        override: Optional[bool] = False,
        tslice=slice(None),
        zslice=slice(None),
        yslice=slice(None),
        xslice=slice(None),
        cslice=2,
    ):
        with h5py.File(self.path, "r") as f:
            arr = da.from_array(f[dset], chunks=(None, None, 16, None, None))
            arr = arr[(tslice, zslice, yslice, xslice, cslice)]  # slice
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
            arr = arr.compute()

        freqs = np.fft.rfftfreq(arr.shape[0], self.dt)
        kvecs = np.fft.fftshift(np.fft.fftfreq(arr.shape[1], self.dx)) * 1e-6
        if name is not None:
            self.add_dset(arr, f"{name}/arr", override)
            self.add_dset(freqs, f"{name}/freqs", override)
            self.add_dset(kvecs, f"{name}/kvecs", override)

        return arr

    def fft(
        self,
        dset: str,
        name: Optional[str] = None,
        override: Optional[bool] = False,
        tslice=slice(None),
        zslice=slice(None),
        yslice=slice(None),
        xslice=slice(None),
        cslice=2,
    ):
        if "dt" not in self.attrs:
            print("Add the 'dt' attribute before calculating the fft")
            return
        if name is not None and f"{name}/arr" in self.dsets and not override:
            raise NameError(
                f"'{name}' already exists, you can use 'override=True' to replace it"
            )
        with h5py.File(self.path, "r") as f:
            arr = da.from_array(f[dset], chunks=(None, None, 16, None, None))
            arr = arr[(tslice, zslice, yslice, xslice, cslice)]
            arr = arr.sum(axis=1)  # sum all z
            arr = da.subtract(arr, arr[0])
            arr = da.subtract(arr, da.average(arr, axis=0)[None, :])
            arr = da.multiply(arr, np.hanning(arr.shape[0])[:, None, None])
            arr = da.swapaxes(arr, 0, -1)
            arr = da.reshape(
                arr, (arr.shape[0] * arr.shape[1], arr.shape[2])
            )  # flatten all the cells
            arr = da.fft.rfft(arr)
            arr = da.absolute(arr)
            arr = da.sum(arr, axis=0)
            arr = arr.compute()

        freqs = np.fft.rfftfreq(self.shape(dset)[0], self.dt)

        if name is not None:
            self.add_dset(arr, f"{name}/arr", override)
            self.add_dset(freqs, f"{name}/freqs", override)

    def fft_tb(
        self, dset: str, tmax: int = None, tmin: int = None, normalize: bool = False
    ):
        y = self[f"table/{dset}"][slice(tmin, tmax)]
        x = np.fft.rfftfreq(y.shape[0], self.dt) * 1e-9
        y -= y[0]
        y -= np.average(y)
        y = np.multiply(y, np.hanning(y.shape[0]))
        y = np.fft.rfft(y)
        y = np.abs(y)
        if normalize:
            y /= y.max()
        return x, y

    def compute_sk_number(self, dset: str, z: int = 0, t: int = 0):
        spin_grid = self[dset][t, z, :, :, :]
        spin_pad = np.pad(
            spin_grid, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0.0
        )
        sk_number = (
            np.cross(
                spin_pad[2:, 1:-1, :],  # s(i+1,j)
                spin_pad[1:-1, 2:, :],  # s(i,j+1)
                axis=2,
            )
            + np.cross(
                spin_pad[:-2, 1:-1, :],  # s(i-1,j)
                spin_pad[1:-1, :-2, :],  # s(i,j-1)
                axis=2,
            )
            - np.cross(
                spin_pad[:-2, 1:-1, :],  # s(i-1,j)
                spin_pad[1:-1, 2:, :],  # s(i,j+1)
                axis=2,
            )
            - np.cross(
                spin_pad[2:, 1:-1, :],  # s(i+1,j)
                spin_pad[1:-1, :-2, :],  # s(i,j-1)
                axis=2,
            )
        )
        sk_number = -1 * np.einsum("ijk,ijk->ij", spin_grid, sk_number) / (16 * np.pi)
        return np.sum(sk_number.flatten())

    # plotting

    def imshow(self, dset: str, zero: bool = True, t: int = -1, c: int = 2, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
        else:
            fig = ax.figure
        if zero:
            arr = self[dset][[0, t], 0, :, :, c]
            arr = arr[1] - arr[0]
        else:
            arr = self[dset][t, 0, :, :, c]
        amin, amax = arr.min(), arr.max()
        if amin < 0 < amax:
            cmap = "cmo.balance"
            vmm = max((-amin, amax))
            vmin, vmax = -vmm, vmm
        else:
            cmap = "cmo.amp"
            vmin, vmax = amin, amax
        ax.imshow(
            arr,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[
                0,
                arr.shape[1] * self.dx * 1e9,
                0,
                arr.shape[0] * self.dy * 1e9,
            ],
        )
        ax.set(
            title=self.name,
            xlabel="x (nm)",
            ylabel="y (nm)",
        )
        fig.colorbar(ax.get_images()[0], ax=ax)

        return ax

    def snapshot_png(self, image: str):
        arr = self[f"snapshots/{image}"][:]
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.imshow(
            arr,
            origin="lower",
            extent=[0, arr.shape[0] * self.dy * 1e9, 0, arr.shape[1] * self.dx * 1e9],
        )
        ax.set_ylabel(r"$y$ (nm)")
        ax.set_xlabel(r"$x$ (nm)")
        fig.tight_layout()

    def snapshot(self, dset: str, z: int = 0, t: int = -1, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
        else:
            fig = ax.figure
        arr = self[dset][t, z, :, :, :]
        arr = np.ma.masked_equal(arr, 0)
        u = arr[:, :, 0]
        v = arr[:, :, 1]
        z = arr[:, :, 2]

        alphas = -np.abs(z) + 1
        hsl = np.ones((u.shape[0], u.shape[1], 3))
        hsl[:, :, 0] = np.angle(u + 1j * v) / np.pi / 2 + 0.5  # normalization
        hsl[:, :, 1] = np.sqrt(u ** 2 + v ** 2 + z ** 2)
        hsl[:, :, 2] = (z + 1) / 2
        rgb = hsl2rgb(hsl)
        stepx = max(int(u.shape[1] / 40), 1)
        stepy = max(int(u.shape[0] / 40), 1)
        scale = 1 / max(stepx, stepy)
        x, y = np.meshgrid(
            np.arange(0, u.shape[1], stepx) * self.dx * 1e9,
            np.arange(0, u.shape[0], stepy) * self.dy * 1e9,
        )
        antidots = np.ma.masked_not_equal(self[dset][0, 0, :, :, 2], 0)
        ax.quiver(
            x,
            y,
            u[::stepy, ::stepx],
            v[::stepy, ::stepx],
            alpha=alphas[::stepy, ::stepx],
            angles="xy",
            scale_units="xy",
            scale=scale,
        )

        ax.imshow(
            rgb,
            interpolation="None",
            origin="lower",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            extent=[
                0,
                arr.shape[1] * self.dx * 1e9,
                0,
                arr.shape[0] * self.dy * 1e9,
            ],
        )
        ax.imshow(
            antidots,
            interpolation="None",
            origin="lower",
            cmap="Set1_r",
            extent=[
                0,
                arr.shape[1] * self.dx * 1e9,
                0,
                arr.shape[0] * self.dy * 1e9,
            ],
        )
        ax.set(title=self.name, xlabel="x (nm)", ylabel="y (nm)")

        L, H = np.mgrid[0 : 1 : arr.shape[1] * 1j, 0:1:20j]
        S = np.ones_like(L)
        rgb = hsl2rgb(np.dstack((H, S, L)))
        fig.tight_layout()
        self.add_radial_phase_colormap(ax)
        return ax

    def add_radial_phase_colormap(self, ax, rec=None):
        def func1(hsl):
            return np.array(colorsys.hls_to_rgb(hsl[0] / (2 * np.pi), hsl[1], hsl[2]))

        if rec is None:
            rec = [0.85, 0.85, 0.15, 0.15]
        cax = plt.axes(rec, polar=True)
        p1, p2 = ax.get_position(), cax.get_position()
        cax.set_position([p1.x1 - p2.width, p1.y1 - p2.height, p2.width, p2.height])

        theta = np.linspace(0, 2 * np.pi, 360)
        r = np.arange(0, 100, 1)
        hls = np.ones((theta.size * r.size, 3))

        hls[:, 0] = np.tile(theta, r.size)
        white_pad = 20
        black_pad = 10
        hls[: white_pad * theta.size, 1] = 1
        hls[-black_pad * theta.size :, 1] = 0
        hls[white_pad * theta.size : -black_pad * theta.size, 1] = np.repeat(
            np.linspace(1, 0, r.size - white_pad - black_pad), theta.size
        )
        rgb = np.apply_along_axis(func1, 1, hls)
        cax.pcolormesh(
            theta,
            r,
            np.zeros((r.size, theta.size)),
            color=rgb,
            shading="nearest",
        )
        cax.spines["polar"].set_visible(False)
        cax.set(yticks=[], xticks=[])
        up_symbol = dict(x=0.5, y=0.5, name=r"$\bigotimes$")
        down_symbol = dict(x=0.1, y=0.5, name=r"$\bigodot$")
        for s in [up_symbol, down_symbol]:
            cax.text(
                s["x"],
                s["y"],
                s["name"],
                color="#3b5bff",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=5,
                transform=cax.transAxes,
            )

    def mode(self, dset: str, f: float, c: int):
        if f"modes/{dset}/arr" not in self.dsets:
            self.calculate_modes(dset)
        fi = (np.abs(self[f"modes/{dset}/freqs"][:] - f)).argmin()
        return self[f"modes/{dset}/arr"][fi, :, :, c]

    def calculate_modes(self, dset: str, z: int = 0, override=False):
        arr = self[dset][:, z, :, :, :]
        s = arr.shape
        arr = arr.reshape(s[0], np.prod(s[1:])).T
        arr = np.fft.rfft(arr)
        arr = arr.T.reshape((int(s[0] / 2 + 1), s[1], s[2], s[3]))

        self.add_dset(arr, f"modes/{dset}/arr", override=override)
        freqs = np.fft.rfftfreq(s[0], self.dt) * 1e-9
        self.add_dset(freqs, f"modes/{dset}/freqs", override=override)

    def calculate_modes2(self, dset: str, z: int = 0, override=False):
        arr = self[dset][:, z, :, :, :]
        arr = np.sqrt(arr[..., 0] ** 2 + arr[..., 1] ** 2 + arr[..., 2] ** 2)
        s = arr.shape
        arr = arr.reshape(s[0], np.prod(s[1:])).T
        arr = np.fft.rfft(arr)
        arr = arr.T.reshape((int(s[0] / 2 + 1), s[1], s[2]))

        self.add_dset(arr, f"modes/{dset}2/arr", override=override)
        freqs = np.fft.rfftfreq(s[0], self.dt) * 1e-9
        self.add_dset(freqs, f"modes/{dset}2/freqs", override=override)

    def plot_mode(self, dset: str, f: float, c: int, axes=None):
        mode = self.mode(dset, f, c)
        max_amp = np.abs(mode).max()

        if axes is None:
            fig, axes = plt.subplots(1, 3, sharey=True)
        (ax0, ax1, ax2) = axes
        fig = ax0.figure
        ax0.imshow(np.abs(mode), cmap="inferno")

        ax1.imshow(
            np.angle(mode), cmap="hsv", vmin=-np.pi, vmax=np.pi, interpolation="None"
        )

        alphas = np.abs(mode) / max_amp
        ax2.imshow(np.angle(mode), alpha=alphas, cmap="hsv", vmin=-np.pi, vmax=np.pi)

        cb = fig.colorbar(ax0.get_images()[0], cax=ax0.inset_axes((1.05, 0.0, 0.05, 1)))
        cb.ax.set_ylabel("Amplitude", rotation=-90, labelpad=8)

        cb = fig.colorbar(
            ax1.get_images()[0],
            cax=ax1.inset_axes((1.05, 0.0, 0.05, 1)),
            ticks=[-3, 0, 3],
        )
        cb.set_ticklabels([r"-$\pi$", 0, r"$\pi$"])
        cb.ax.set_ylabel("Phase", rotation=-90)
        cb = fig.colorbar(
            ax1.get_images()[0],
            cax=ax2.inset_axes((1.05, 0.0, 0.05, 1)),
            ticks=[-3, 0, 3],
        )
        cb.set_ticklabels([r"-$\pi$", 0, r"$\pi$"])
        cb.ax.set_ylabel("Phase", rotation=-90)

        # ax0.set_title(f"Comp = {c}")
        # ax1.set_title(f"Amps = {max_amp:.3f}")
        # ax2.set_title(f"f = {ff:.3f} GHz")

        # fig.suptitle(self.name, pad=0)
        # fig.tight_layout(h_pad=0, w_pad=0)

    def plot_modes(self, dset: str, f: float):
        mode1 = self.mode(dset, f, 0)
        mode2 = self.mode(dset, f, 1)
        mode3 = self.mode(dset, f, 2)
        modes = np.array([mode1, mode2, mode3])
        modes_max = np.abs(modes).max()
        extent = [0, mode1.shape[0] * self.dy * 1e9, 0, mode1.shape[1] * self.dx * 1e9]

        fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4, 4))
        for c in range(3):
            mode_abs = np.abs(modes[c])
            mode_ang = np.angle(modes[c])
            alphas = mode_abs / mode_abs.max()

            axes[0, c].imshow(
                mode_abs, cmap="inferno", vmin=0, vmax=modes_max, extent=extent
            )
            axes[1, c].imshow(
                mode_ang,
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="None",
                extent=extent,
            )
            axes[2, c].imshow(
                mode_ang,
                alpha=alphas,
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="None",
                extent=extent,
            )
        axes[0, 0].set_title(r"$m_x$")
        axes[0, 1].set_title(r"$m_y$")
        axes[0, 2].set_title(r"$m_z$")
        axes[0, 0].set_ylabel(r"$y$ (nm)")
        axes[1, 0].set_ylabel(r"$y$ (nm)")
        axes[2, 0].set_ylabel(r"$y$ (nm)")
        axes[2, 0].set_xlabel(r"$x$ (nm)")
        axes[2, 1].set_xlabel(r"$x$ (nm)")
        axes[2, 2].set_xlabel(r"$x$ (nm)")
        cb = fig.colorbar(
            axes[0, 2].get_images()[0], cax=axes[0, 2].inset_axes((1.05, 0.0, 0.05, 1))
        )
        cb.ax.set_ylabel("Amplitude")
        for i in [1, 2]:
            cb = fig.colorbar(
                axes[1, 2].get_images()[0],
                cax=axes[i, 2].inset_axes((1.05, 0.0, 0.05, 1)),
                ticks=[-3, 0, 3],
            )
            cb.set_ticklabels([r"-$\pi$", 0, r"$\pi$"])
            cb.ax.set_ylabel("Phase")
        fi = (np.abs(self[f"modes/{dset}/freqs"][:] - f)).argmin()
        ff = self[f"modes/{dset}/freqs"][:][fi]
        fig.suptitle(f"Simulation: {self.name}/{dset}, Frequency: {ff:.2f} GHz")
        fig.tight_layout()

    def plot_modes2(self, dset: str, f: float):
        fi = (np.abs(self[f"modes/{dset}/freqs"][:] - f)).argmin()
        ff = self[f"modes/{dset}/freqs"][:][fi]
        mode = self[f"modes/{dset}/arr"][fi, :, :]
        extent = [0, mode.shape[0] * self.dy * 1e9, 0, mode.shape[1] * self.dx * 1e9]

        fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 2))
        mode_abs = np.abs(mode)
        mode_ang = np.angle(mode)
        alphas = mode_abs / mode_abs.max()

        axes[0].imshow(
            mode_abs, cmap="inferno", vmin=0, vmax=mode_abs.max(), extent=extent
        )
        axes[1].imshow(
            mode_ang,
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="None",
            extent=extent,
        )
        axes[2].imshow(
            mode_ang,
            alpha=alphas,
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="None",
            extent=extent,
        )
        axes[0].set_ylabel(r"$y$ (nm)")
        axes[0].set_xlabel(r"$x$ (nm)")
        axes[1].set_xlabel(r"$x$ (nm)")
        axes[2].set_xlabel(r"$x$ (nm)")
        cb = fig.colorbar(
            axes[0].get_images()[0], cax=axes[0].inset_axes((1.05, 0.0, 0.05, 1))
        )
        cb.ax.set_ylabel("Amplitude")
        for i in [1, 2]:
            cb = fig.colorbar(
                axes[1].get_images()[0],
                cax=axes[i].inset_axes((1.05, 0.0, 0.05, 1)),
                ticks=[-3, 0, 3],
            )
            cb.set_ticklabels([r"-$\pi$", 0, r"$\pi$"])
            cb.ax.set_ylabel("Phase")
        fig.suptitle(f"Simulation: {self.name}, Frequency: {ff:.2f} GHz, sum of comps")
        fig.tight_layout()

    def plot_fft_tb(
        self, tmin=0, tmax=-1, fft_tmin=0, fft_tmax=-1, thres=0.15, axes=None
    ):
        if axes is None:
            _, axes = plt.subplots(3, 1, sharex=True)
        comps = ["mx", "my", "mz"]
        fss = []
        for i in range(3):
            c, ax = comps[i], axes[i]
            x, y = self.fft_tb(c, tmax=fft_tmax, tmin=fft_tmin)
            ax.plot(x[tmin:tmax], y[tmin:tmax])
            list_peaks = peakutils.indexes(y, thres=thres, min_dist=4)
            fs = [x[i] for i in list_peaks]
            fss.append(fs)
            for f in fs:
                ax.axvline(f, ls="--", c="gray")
                ax.text(
                    f,
                    y[tmin:].max() * 1.15,
                    f"{f:.2f}",
                    fontsize=5,
                    rotation=45,
                    ha="left",
                    va="center",
                )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # fig.tight_layout()
        return axes, fss

    def plot_anim(self, dset: str, z: int = 0, tmin: int = 0, tmax: int = -1):
        arr = self[dset][tmin:tmax, z, :, :, :]
        # arr -= arr[0]
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
        # Arrays
        arr = np.ma.masked_equal(arr, 0)
        u, v, z = arr[..., 0], arr[..., 1], arr[..., 2]
        alphas = -np.abs(z) + 1
        hsl = np.ones((u.shape[0], u.shape[1], u.shape[2], 3))
        hsl[..., 0] = np.angle(u + 1j * v) / np.pi / 2 + 0.5  # normalization
        hsl[..., 1] = np.sqrt(u ** 2 + v ** 2 + z ** 2)
        hsl[..., 2] = (z + 1) / 2
        rgb = hsl2rgb(hsl)
        stepx = max(int(u.shape[2] / 40), 1)
        stepy = max(int(u.shape[1] / 40), 1)
        scale = 1 / max(stepx, stepy)
        x, y = np.meshgrid(
            np.arange(0, u.shape[2], stepx) * self.dx * 1e9,
            np.arange(0, u.shape[1], stepy) * self.dy * 1e9,
        )
        antidots = np.ma.masked_not_equal(self["m"][0, 0, :, :, 2], 0)
        extent = [0, arr.shape[2] * self.dx * 1e9, 0, arr.shape[1] * self.dy * 1e9]
        t = 0
        Q = ax.quiver(
            x,
            y,
            u[t, ::stepy, ::stepx],
            v[t, ::stepy, ::stepx],
            alpha=alphas[t, ::stepy, ::stepx],
            angles="xy",
            scale_units="xy",
            scale=scale,
        )

        ax.imshow(
            rgb[t],
            interpolation="None",
            origin="lower",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            extent=extent,
        )
        ax.imshow(
            antidots, interpolation="None", origin="lower", cmap="Set1_r", extent=extent
        )
        ax.get_images()[0].set_data(rgb[t + 1])
        ax.set(title=self.name, xlabel="x (nm)", ylabel="y (nm)")
        fig.tight_layout()

        def run(t):
            ax.get_images()[0].set_data(rgb[t])
            Q.set_UVC(u[t, ::stepy, ::stepx], v[t, ::stepy, ::stepx])
            Q.set_alpha(alphas[t, ::stepy, ::stepx])
            ax.set_title(f"t={t}")
            return ax

        ani = mpl.animation.FuncAnimation(
            fig, run, interval=1, frames=np.arange(1, arr.shape[0], dtype="int")
        )
        anim_save_path = self.path.replace(".h5", ".mp4")
        ani.save(
            anim_save_path,
            writer="ffmpeg",
            fps=25,
            dpi=300,
            extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"],
        )
        print(f"Saved at: {anim_save_path}")
