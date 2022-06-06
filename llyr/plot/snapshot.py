import colorsys

import matplotlib.pyplot as plt
import numpy as np

from .._utils import hsl2rgb, add_radial_phase_colormap
from ..base import Base


class snapshot(Base):
    def plot(
        self, dset: str = "m", z: int = 0, t: int = -1, ax=None, repeat=1, zero=None
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
        else:
            fig = ax.figure
        arr = self.m[dset][t, z, :, :, :]
        if zero is not None:
            arr -= self.m[dset][zero, z, :, :, :]
        arr = np.tile(arr, (repeat, repeat, 1))
        arr = np.ma.masked_equal(arr, 0)
        u = arr[:, :, 0]
        v = arr[:, :, 1]
        z = arr[:, :, 2]

        alphas = -np.abs(z) + 1
        hsl = np.ones((u.shape[0], u.shape[1], 3))
        hsl[:, :, 0] = np.angle(u + 1j * v) / np.pi / 2  # normalization
        hsl[:, :, 1] = np.sqrt(u**2 + v**2 + z**2)
        hsl[:, :, 2] = (z + 1) / 2
        rgb = hsl2rgb(hsl)
        stepx = max(int(u.shape[1] / 60), 1)
        stepy = max(int(u.shape[0] / 60), 1)
        scale = 1 / max(stepx, stepy)
        x, y = np.meshgrid(
            np.arange(0, u.shape[1], stepx) * self.m.dx * 1e9,
            np.arange(0, u.shape[0], stepy) * self.m.dy * 1e9,
        )
        antidots = np.ma.masked_not_equal(self.m[dset][0, 0, :, :, 2], 0)
        antidots = np.tile(antidots, (repeat, repeat))
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
                rgb.shape[1] * self.m.dx * 1e9,
                0,
                rgb.shape[0] * self.m.dy * 1e9,
            ],
        )
        ax.imshow(
            antidots,
            interpolation="None",
            origin="lower",
            cmap="Set1_r",
            extent=[
                0,
                arr.shape[1] * self.m.dx * 1e9,
                0,
                arr.shape[0] * self.m.dy * 1e9,
            ],
        )
        ax.set(title=self.m.sim_name, xlabel="x (nm)", ylabel="y (nm)")
        add_radial_phase_colormap(ax)
        return ax
