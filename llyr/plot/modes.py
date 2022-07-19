from configparser import Interpolation
import matplotlib.pyplot as plt
import numpy as np

from .._utils import hsl2rgb

from ..base import Base


class modes(Base):
    def plot(self, dset: str, f: float, z: int = 0, axes=None):
        mode_list = self.m.get_mode(dset, f)[z]
        fig = plt.figure(figsize=(6, 6), dpi=140)
        gs = fig.add_gridspec(
            3, 3, left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01
        )
        axes = np.array(
            [[fig.add_subplot(gs[i, j]) for i in range(3)] for j in range(3)]
        )
        mode_list_max = np.abs(mode_list).max()

        for c in range(3):
            mode_abs = np.abs(mode_list[..., c])
            mode_ang = np.angle(mode_list[..., c])
            alphas = mode_abs / mode_abs.max()
            axes[c, 2].imshow(
                mode_abs,
                cmap="inferno",
                vmin=0,
                vmax=mode_list_max,
                aspect="equal",
            )
            axes[c, 1].imshow(
                mode_ang,
                aspect="equal",
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="None",
            )
            axes[c, 0].imshow(
                mode_ang,
                aspect="equal",
                alpha=alphas,
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="None",
            )
            # axes[2, c].set_aspect(1)
        for ax in axes.flatten():
            ax.set_frame_on(False)
            ax.set(xticks=[], yticks=[])

    def plot_one(
        self,
        ax: plt.Axes,
        dset: str,
        f: float,
        comp: int,
        color: str,
        z: int = 0,
        repeat: int = 1,
    ):
        mode = self.m.get_mode(dset, f, comp)[z]
        mode = np.tile(mode, (repeat, repeat))
        extent = [
            0,
            mode.shape[1] * self.m.dx * 1e9,
            0,
            mode.shape[0] * self.m.dy * 1e9,
        ]
        if color == "amp":
            ax.imshow(
                np.abs(mode),
                cmap="inferno",
                vmin=0,
                extent=extent,
                aspect="equal",
            )
        elif color == "phase":
            ax.imshow(
                np.angle(mode),
                aspect="equal",
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="None",
                extent=extent,
            )
        elif color == "phaseamp":
            ax.imshow(
                np.angle(mode),
                aspect="equal",
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                extent=extent,
                alpha=np.abs(mode) / np.abs(mode).max(),
                interpolation="None",
            )
        else:
            raise ValueError(
                "Invalid 'color' argument, possible values are: ['amp','phase','phaseamp']"
            )

    def plot_one_v2(self, ax, dset, f, comp, repeat=1, z=0):
        arr = self.m.get_mode(dset, f)[z]
        arr = np.real(arr)
        arr = np.tile(arr, (repeat, repeat))
        arr = np.ma.masked_equal(arr, 0)
        u, v, w = arr[..., 0], arr[..., 1], arr[..., 2]
        alphas = np.abs(w) / np.abs(w).max()
        hsl = np.ones((u.shape[0], u.shape[1], 3))
        hsl[..., 0] = np.angle(u + 1j * v) / np.pi / 2  # normalization
        hsl[..., 1] = np.sqrt(u**2 + v**2 + w**2)
        hsl[..., 2] = (w + 1) / 2
        rgb = hsl2rgb(hsl)
        stepx = max(int(u.shape[1] / 60), 1)
        stepy = max(int(u.shape[0] / 60), 1)
        scale = 1 / max(stepx, stepy) * 10
        x, y = np.meshgrid(
            np.arange(0, u.shape[1], stepx) * self.m.dx * 1e9,
            np.arange(0, u.shape[0], stepy) * self.m.dy * 1e9,
        )
        # antidots = np.ma.masked_not_equal(self.m["m"][0, 0, :, :, 2], 0)
        # antidots = np.tile(antidots, (repeat, repeat))
        extent = [
            0,
            arr.shape[1] * self.m.dx * 1e9,
            0,
            arr.shape[0] * self.m.dy * 1e9,
        ]
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
            extent=extent,
            alpha=alphas,
        )
        # ax.imshow(
        #     antidots, interpolation="None", origin="lower", cmap="Set1_r", extent=extent
        # )
        ax.set(xticks=[], yticks=[])
