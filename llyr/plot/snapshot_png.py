import matplotlib.pyplot as plt

from ..base import Base


class snapshot_png(Base):
    def plot(self, image: str):
        arr = self.llyr[f"snapshots/{image}"][:]
        fig, ax = plt.subplots(1, 1, figsize=(4, 2))
        ax.imshow(
            arr,
            origin="lower",
            extent=[
                0,
                arr.shape[0] * self.llyr.dy * 1e9,
                0,
                arr.shape[1] * self.llyr.dx * 1e9,
            ],
        )
        ax.set_ylabel(r"$y$ (nm)")
        ax.set_xlabel(r"$x$ (nm)")
        fig.tight_layout()
        return self