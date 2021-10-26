import matplotlib.pyplot as plt
import numpy as np

from ..base import Base


class modes(Base):
    def plot(self, dset: str, f: float, axes=None):
        mode1 = self.llyr.calc.mode(dset, f, 0)
        mode2 = self.llyr.calc.mode(dset, f, 1)
        mode3 = self.llyr.calc.mode(dset, f, 2)
        mode_list = np.array([mode1, mode2, mode3])
        mode_list_max = np.abs(mode_list).max()
        extent = [
            0,
            mode1.shape[1] * self.llyr.dy * 1e9,
            0,
            mode1.shape[0] * self.llyr.dx * 1e9,
        ]

        if axes is None:
            fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4, 4))
        else:
            fig = axes[0, 0].figure
        for c in range(3):
            mode_abs = np.abs(mode_list[c])
            mode_ang = np.angle(mode_list[c])
            alphas = mode_abs / mode_abs.max()

            axes[0, c].imshow(
                mode_abs,
                cmap="inferno",
                vmin=0,
                vmax=mode_list_max,
                extent=extent,
                aspect="equal",
            )
            axes[1, c].imshow(
                mode_ang,
                aspect="equal",
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="None",
                extent=extent,
            )
            axes[2, c].imshow(
                mode_ang,
                aspect="equal",
                alpha=alphas,
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="None",
                extent=extent,
            )
        # axes[0, 0].set_title(r"$m_x$")
        # axes[0, 1].set_title(r"$m_y$")
        # axes[0, 2].set_title(r"$m_z$")
        # axes[0, 0].set_ylabel(r"$y$ (nm)")
        # axes[1, 0].set_ylabel(r"$y$ (nm)")
        # axes[2, 0].set_ylabel(r"$y$ (nm)")
        # axes[2, 0].set_xlabel(r"$x$ (nm)")
        # axes[2, 1].set_xlabel(r"$x$ (nm)")
        # axes[2, 2].set_xlabel(r"$x$ (nm)")
        # cb = fig.colorbar(
        #     axes[0, 2].get_images()[0], cax=axes[0, 2].inset_axes((1.05, 0.0, 0.05, 1))
        # )
        # cb.ax.set_ylabel("Amplitude")
        # for i in [1, 2]:
        #     cb = fig.colorbar(
        #         axes[1, 2].get_images()[0],
        #         cax=axes[i, 2].inset_axes((1.05, 0.0, 0.05, 1)),
        #         ticks=[-3, 0, 3],
        #     )
        #     cb.set_ticklabels([r"-$\pi$", 0, r"$\pi$"])
        #     cb.ax.set_ylabel("Phase")
        # fi = (np.abs(self.llyr[f"mode_list/{dset}/freqs"][:] - f)).argmin()
        # ff = self.llyr[f"mode_list/{dset}/freqs"][:][fi]
        fig.suptitle(f"{self.llyr.name}")
        # fig.tight_layout()
        for ax in axes.flatten():
            ax.set(xticks=[], yticks=[])
        return self
