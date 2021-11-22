import colorsys

import matplotlib.pyplot as plt
import numpy as np

from .._utils import hsl2rgb
from ..base import Base


class snapshot(Base):
    def plot(self, dset: str = "m", z: int = 0, t: int = -1, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
        else:
            fig = ax.figure
        arr = self.llyr[dset][t, z, :, :, :]
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
        stepx = max(int(u.shape[1] / 60), 1)
        stepy = max(int(u.shape[0] / 60), 1)
        scale = 1 / max(stepx, stepy)
        x, y = np.meshgrid(
            np.arange(0, u.shape[1], stepx) * self.llyr.dx * 1e9,
            np.arange(0, u.shape[0], stepy) * self.llyr.dy * 1e9,
        )
        antidots = np.ma.masked_not_equal(self.llyr[dset][0, 0, :, :, 2], 0)
        print(x.shape, y.shape, u.shape, v.shape)
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
        ax.set(title=self.llyr.aname, xlabel="x (nm)", ylabel="y (nm)")
        L, H = np.mgrid[0 : 1 : arr.shape[1] * 1j, 0:1:20j]  # type: ignore
        S = np.ones_like(L)
        rgb = hsl2rgb(np.dstack((H, S, L)))
        fig.tight_layout()
        self.add_radial_phase_colormap(ax)
        return self

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
