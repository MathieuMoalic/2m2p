import time
import k3d
import numpy as np

from ..base import Base
from .._utils import rgb_int_from_vectors, normalize


class p3d(Base):
    def plot(
        self,
        dset: str = "m",
        tslice=None,
        zslice=None,
        yslice=None,
        xslice=None,
        data=None,
        norm=True,
    ):
        if data is None:
            if tslice is None:
                tslice = slice(-1, None)
            if zslice is None:
                zslice = slice(0, 1)
            if yslice is None:
                yslice = slice(None, None, 10)
            if xslice is None:
                xslice = slice(None, None, 10)
            arr = self.m[dset][tslice, zslice, yslice, xslice]
        else:
            if data.ndim == 5:
                arr = data
            elif data.ndim == 4:
                arr = data[None, ...]
            else:
                raise ValueError(f"Wrong data shape: {data.shape}")
        if norm:
            arr = normalize(arr)
        space_size = np.prod(arr.shape[1:-1])
        pre_colors = rgb_int_from_vectors(arr)
        origins = np.zeros((arr.shape[0], space_size, 3), dtype=np.float32)
        vectors = np.zeros((arr.shape[0], space_size, 3), dtype=np.float32)
        colors = np.zeros((arr.shape[0], space_size), dtype=np.uint32)
        for it in range(arr.shape[0]):
            i = 0
            for iz in range(arr.shape[1]):
                for iy in range(arr.shape[2]):
                    for ix in range(arr.shape[3]):
                        # print(arr[it, iz, iy, ix])
                        if (arr[it, iz, iy, ix] != np.zeros(3)).any():
                            origins[it, i] = (ix, iy, iz)
                            vectors[it, i] = arr[it, iz, iy, ix]
                            colors[it, i] = pre_colors[it, iz, iy, ix]
                            i += 1
        colors = np.repeat(colors, 2).reshape(arr.shape[0], space_size * 2)
        origins = origins[:, :i]
        vectors = vectors[:, :i]
        colors = colors[:, : i * 2]
        fig = k3d.plot()
        fig.background_color = 0x4F4F4F
        fig.grid_visible = False
        ax = k3d.vectors(
            origins=origins[0] / 2 - vectors[0] / 2,
            vectors=vectors[0],
            colors=colors[0],
            line_width=1e-1,
            head_size=3,
        )
        fig += ax
        fig.display()
        self.o = origins
        self.v = vectors
        self.c = colors
        self.fig = fig
        self.ax = ax

        if arr.ndim == 5:
            self.run(5)
        return self

    def run(self, periods, fps=15):
        ts = self.o.shape[0]
        for t in range(ts * periods):
            self.ax.origins = self.o[t % ts] / 2 - self.v[t % ts] / 2
            self.ax.vectors = self.v[t % ts]
            self.ax.colors = self.c[t % ts]
            time.sleep(1 / fps)
