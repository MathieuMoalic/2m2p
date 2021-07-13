import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean  # pylint: disable=unused-import
import numpy as np
from colorconversion import hsl2rgb  # type: ignore


class Plot:
    def __init__(self, llyr) -> None:
        self.llyr = llyr

    def imshow(self, dset: str, zero: bool = True, t: int = -1, c: int = 2, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=200)
        else:
            fig = ax.figure
        if zero:
            arr = self.llyr[dset][[0, t], 0, :, :, c]
            arr = arr[1] - arr[0]
        else:
            arr = self.llyr[dset][t, 0, :, :, c]
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
                arr.shape[1] * self.llyr.dx * 1e9,
                0,
                arr.shape[0] * self.llyr.dy * 1e9,
            ],
        )
        ax.set(
            title=self.llyr.name,
            xlabel="x (nm)",
            ylabel="y (nm)",
        )
        fig.colorbar(ax.get_images()[0], ax=ax)

        return ax

    def cimshow(self, dset: str, zero: bool = False, t: int = -1, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=200)
        else:
            fig = ax.figure
        if zero:
            arr = self.llyr[dset][[0, t], 0, :, :, :]
            arr = arr[1] - arr[0]
        else:
            arr = self.llyr[dset][t, 0, :, :, :]
        arr = np.ma.masked_equal(arr, 0)
        u = arr[:, :, 0]
        v = arr[:, :, 1]
        z = arr[:, :, 2]

        alphas = -np.abs(z) + 1
        hsl = np.ones((u.shape[0], u.shape[1], 3))
        hsl[:, :, 0] = np.angle(u + 1j * v) * 180 / np.pi
        hsl[:, :, 1] = np.sqrt(u ** 2 + v ** 2 + z ** 2)
        hsl[:, :, 2] = (z + 1) / 2
        rgb = hsl2rgb(hsl)
        stepx = int(u.shape[1] / 40)
        stepy = int(u.shape[1] / 40)
        x, y = np.meshgrid(
            np.arange(0, u.shape[1], stepx) * self.llyr.dx * 1e9,
            np.arange(0, u.shape[0], stepy) * self.llyr.dy * 1e9,
        )
        antidots = np.ma.masked_not_equal(self.llyr[dset][0, 0, :, :, 2], 0)
        print(np.max(rgb), np.min(rgb))
        ax.quiver(
            x,
            y,
            u[::stepy, ::stepx],
            v[::stepy, ::stepx],
            alpha=alphas[::stepy, ::stepx],
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
                arr.shape[1] * self.llyr.dx * 1e9,
                0,
                arr.shape[0] * self.llyr.dy * 1e9,
            ],
        )
        ax.imshow(
            antidots,
            interpolation="None",
            origin="lower",
            cmap="Set1_r",
            extent=[
                0,
                arr.shape[1] * self.llyr.dx * 1e9,
                0,
                arr.shape[0] * self.llyr.dy * 1e9,
            ],
        )
        ax.set(title=self.llyr.name, xlabel="x (nm)", ylabel="y (nm)")
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(mpl.colors.Normalize(-np.pi, np.pi), "hsv"),
            ax=ax,
            cax=ax.inset_axes((1.05, 0.0, 0.05, 1)),
            ticks=[-np.pi, 0, np.pi],
        )
        cb.ax.set_yticklabels([r"-$\pi$", 0, r"$\pi$"])
