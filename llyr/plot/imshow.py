import matplotlib.pyplot as plt

from ..base import Base


class imshow(Base):
    def plot(self, dset: str, zero: bool = True, t: int = -1, c: int = 2, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=200)
        else:
            fig = ax.figure
        if zero:
            arr = self.m[dset][[0, t], 0, :, :, c]
            arr = arr[1] - arr[0]
        else:
            arr = self.m[dset][t, 0, :, :, c]
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
                arr.shape[1] * self.m.dx * 1e9,
                0,
                arr.shape[0] * self.m.dy * 1e9,
            ],
        )
        ax.set(
            title=self.m.sim_name,
            xlabel="x (nm)",
            ylabel="y (nm)",
        )
        fig.colorbar(ax.get_images()[0], ax=ax)

        return self
