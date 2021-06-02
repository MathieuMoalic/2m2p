import matplotlib.pyplot as plt
import cmocean  # pylint: disable=unused-import


class Plot:
    def __init__(self, llyr) -> None:
        self.llyr = llyr

    def imshow(self, dset: str, zero: bool = True, t: int = -1, c: int = 2):
        fig, ax = plt.subplots(1, 1, dpi=200)
        if zero:
            arr = self.llyr[dset][[0, t], 0, :, :, c]
            arr = arr[1] - arr[0]
        else:
            arr = self.llyr[dset][t, 0, :, :, c]
        amin, amax = arr.min(), arr.max()
        if amin < 0 and amax > 0:
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
