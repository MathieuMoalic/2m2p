import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from .._utils import hsl2rgb

from ..base import Base


class anim(Base):
    def plot(self, dset: str, z: int = 0, tmin: int = 0, tmax: int = -1):
        arr = self.llyr[dset][tmin:tmax, z, :, :, :]
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
        stepx = max(int(u.shape[2] / 60), 1)
        stepy = max(int(u.shape[1] / 60), 1)
        scale = 1 / max(stepx, stepy)
        x, y = np.meshgrid(
            np.arange(0, u.shape[2], stepx) * self.llyr.dx * 1e9,
            np.arange(0, u.shape[1], stepy) * self.llyr.dy * 1e9,
        )
        antidots = np.ma.masked_not_equal(self.llyr["m"][0, 0, :, :, 2], 0)
        extent = [
            0,
            arr.shape[2] * self.llyr.dx * 1e9,
            0,
            arr.shape[1] * self.llyr.dy * 1e9,
        ]
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
        ax.set(title=self.llyr.name, xlabel="x (nm)", ylabel="y (nm)")
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
        anim_save_path = self.llyr.path.replace(".h5", ".mp4")
        ani.save(
            anim_save_path,
            writer="ffmpeg",
            fps=25,
            dpi=300,
            extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"],
        )
        print(f"Saved at: {anim_save_path}")
        plt.close()
        return self