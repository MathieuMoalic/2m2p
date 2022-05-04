import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .._utils import hsl2rgb

from ..base import Base


class anim2(Base):
    def get_trgba(arr):
        u, v, z = arr[..., 0], arr[..., 1], arr[..., 2]
        hsl = np.ones((u.shape[0], u.shape[1], u.shape[2], 3))
        hsl[..., 0] = np.angle(u + 1j * v) / np.pi / 2  # normalization
        hsl[..., 1] = 1
        hsl[..., 2] = (z + 1) / 2
        trgba = np.ones((u.shape[0], u.shape[1], u.shape[2], 4))
        trgba[..., :3] = hsl2rgb(hsl)
        trgba[..., 3] = -np.abs(z) + 1
        return trgba

    def get_quiver_data(arr, dx, dy):
        stepx = max(int(arr.shape[2] / 60), 1)
        stepy = max(int(arr.shape[1] / 60), 1)
        scale = 1 / max(stepx, stepy)
        x, y = np.meshgrid(
            np.arange(0, arr.shape[2], stepx) * dx * 1e9,
            np.arange(0, arr.shape[1], stepy) * dy * 1e9,
        )
        u, v, z = (
            arr[:, ::stepy, ::stepx, 0],
            arr[:, ::stepy, ::stepx, 1],
            arr[:, ::stepy, ::stepx, 2],
        )
        alpha = -np.abs(z) + 1
        return x, y, u, v, alpha, scale

    def plot(self, mult=0.9):
        m = self.llyr
        arr = m.m[:60, 0] - m.stable[0, 0] * mult
        arr = np.tile(arr, (1, 2, 2, 1))
        antidots = np.ma.masked_not_equal(m["m"][0, 0, :, :, 2], 0)
        antidots = np.tile(antidots, (2, 2))
        arr = np.ma.masked_equal(arr, 0)
        arr /= np.linalg.norm(arr, axis=-1)[..., None]
        fig = plt.figure(figsize=(8, 8), dpi=200)
        gs = fig.add_gridspec(1, 1, left=0, right=1, top=1, bottom=0)
        ax = fig.add_subplot(gs[:])
        # gs.update()
        arr = np.ma.masked_equal(arr, 0)
        extent = [0, arr.shape[2] * m.dx * 1e9, 0, arr.shape[1] * m.dy * 1e9]
        trgba = self.get_trgba(arr)
        ax.imshow(trgba[0], interpolation="None", origin="lower", extent=extent)
        ax.imshow(
            antidots, interpolation="None", origin="lower", cmap="Set1_r", extent=extent
        )
        x, y, u, v, alpha, scale = self.get_quiver_data(arr, m.dx, m.dy)
        Q = ax.quiver(
            x, y, u[0], v[0], alpha=alpha[0], angles="xy", scale_units="xy", scale=scale
        )
        ts = np.array(m.m.attrs["t"][:]) * 1e12
        ts -= ts[0]
        ax.set_title(f"{ts[0]:.0f} ps")
        ax.set(xticks=[], yticks=[])

        def run(t):
            ax.get_images()[0].set_data(trgba[t])
            ax.set_title(f"{ts[t]:.0f} ps")
            Q.set_UVC(u[t], v[t])
            Q.set_alpha(alpha[t])

        ani = FuncAnimation(
            fig, run, interval=50, frames=np.arange(1, trgba.shape[0], dtype="int")
        )
        ani.save(f"jobs/anim/{m.sim_name}_{mult}.gif", writer="ffmpeg", fps=25, dpi=150)
        plt.close()
