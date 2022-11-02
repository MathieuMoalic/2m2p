import matplotlib.pyplot as plt

from ..base import Base


class disp(Base):
    def plot(self, dset="m", ax=None, slices=(slice(None), slice(None), 0)):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        else:
            fig = ax.figure
        arr = self.m[f"disp/{dset}/disp"][slices]
        freqs = self.m[f"disp/{dset}/freqs"][slices[0]]
        kvecs = self.m[f"disp/{dset}/kvecs"][slices[1]]
        im = ax.imshow(
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
        ax.set_xlabel(r"$k_x$ (1/nm)")
        ax.set_ylabel("f (GHz)")
        ax.set_title(self.m.sim_name)
        ax.set_xlim(-1 / self.m.dx * 1e-9 / 4, 1 / self.m.dx * 1e-9 / 4)
        ax.set_ylim(4, 18)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return ax
