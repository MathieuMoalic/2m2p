import matplotlib.pyplot as plt
import numpy as np

from ..base import Base


class idisp(Base):
    def plot(self, dset="m", slices=(slice(None), slice(None), 0)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        arr = self.m[f"disp/{dset}/disp"][slices]
        freqs = self.m[f"disp/{dset}/freqs"][slices[0]]
        kvecs = self.m[f"disp/{dset}/kvecs"][slices[1]]
        im = ax1.imshow(
            arr,
            aspect="auto",
            origin="lower",
            cmap="cmo.amp_r",
            # norm=mpl.colors.LogNorm(vmin=1e-5),
            extent=[
                kvecs.min() * 1e-9,
                kvecs.max() * 1e-9,
                freqs.min() * 1e-9,
                freqs.max() * 1e-9,
            ],
        )
        ax1.set_xlabel(r"$k_x$ (1/nm)")
        ax1.set_ylabel("f (GHz)")
        ax1.set_title(self.m.sim_name)
        ax1.set_xlim(-1 / self.m.dx * 1e-9 / 5, 1 / self.m.dx * 1e-9 / 5)
        ax1.set_ylim(4, 18)
        fig.colorbar(im, ax=ax1)
        hline = ax1.axhline(5, ls="--", lw=0.8, c="#ffb86c")
        vline = ax1.axvline(0, ls="--", lw=0.8, c="#ffb86c")

        def plot_mode(k, f):
            vline.set_data([k, k], [0, 1])
            hline.set_data([0, 1], [f, f])
            f_idx = np.abs(freqs * 1e-9 - f).argmin()
            kmin = kvecs.shape[0] // 2 - 200
            kmax = kvecs.shape[0] // 2 + 200
            arr = self.m[f"disp/{dset}/fft2d"][:, :, :, 0]
            arr2 = np.fft.ifft2(arr)[1000 + f_idx, kmin:kmax]
            ax2.cla()
            im = ax2.imshow(np.abs(arr2), aspect="auto", origin="lower", zorder=-1)
            cax = ax2.inset_axes(
                [0.6, 0.05, 0.3, 0.04], transform=ax2.transAxes, zorder=10
            )
            fig.colorbar(im, ax=ax2, cax=cax, orientation="horizontal")
            fig.canvas.draw()

        def onclick(event):
            if event.inaxes == ax1:
                plot_mode(event.xdata, event.ydata)

        fig.tight_layout()
        fig.canvas.mpl_connect("button_press_event", onclick)
        return (ax1, ax2)
