import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from ..base import Base


class modes(Base):
    def plot(self, dset: str, f: float, z: int = 0, axes=None):
        mode_list = self.llyr.get_mode(dset, f)[z]
        mode_list_max = np.abs(mode_list).max()
        extent = [
            0,
            mode_list.shape[1] * self.llyr.dx * 1e9,
            0,
            mode_list.shape[0] * self.llyr.dy * 1e9,
        ]

        if axes is None:
            fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(4, 4))
        else:
            fig = axes[0, 0].figure
        for c in range(3):
            mode_abs = np.abs(mode_list[..., c])
            mode_ang = np.angle(mode_list[..., c])
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
            axes[2, c].pcolormesh(
                mode_ang,
                # aspect="equal",
                alpha=alphas,
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                # interpolation="None",
                # extent=extent,
            )
            # axes[2, c].pcolormesh(arr, alpha=arr)
            axes[2, c].set_aspect(1)
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
        # fi = (np.abs(self.llyr[f"mode_list/{dset}/freqs"][:] - f)).argmin()
        # ff = self.llyr[f"mode_list/{dset}/freqs"][:][fi]
        fig.suptitle(f"{self.llyr.sim_name}")
        fig.tight_layout()
        # for ax in axes.flatten():
        #     ax.set(xticks=[], yticks=[])
        self.fig = fig
        return self

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
        mode = self.llyr.get_mode(dset, f, comp)[z]
        mode = np.tile(mode, (repeat, repeat))
        extent = [
            0,
            mode.shape[1] * self.llyr.dx * 1e9,
            0,
            mode.shape[0] * self.llyr.dy * 1e9,
        ]
        if color == "amp":
            ax.imshow(
                np.abs(mode)/np.abs(mode).max(),
                cmap="inferno",
                vmin=0,
                extent=extent,
                aspect="equal",
                norm=mpl.colors.LogNorm(),
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
            )
        else:
            raise ValueError(
                "Invalid 'color' argument, possible values are: ['amp','phase','phaseamp']"
            )
