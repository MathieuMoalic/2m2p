import os
from pathlib import Path
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import zarr

from .plot import Plot
from .calc import Calc

# from ._interactive import iplot
from ._utils import (
    h5_to_zarr,
    load_ovf,
    merge_table,
    get_ovf_parms,
    out_to_zarr,
    hsl2rgb,
)

__all__ = [
    "h5_to_zarr",
    "load_ovf",
    "merge_table",
    "get_ovf_parms",
    "out_to_zarr",
    "hsl2rgb",
    "iplot",
]


def op(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Wrong path.")
    if "ssh://" in path:
        return Group(zarr.storage.FSStore(path))
    else:
        return Group(zarr.storage.DirectoryStore(path))


class Group(zarr.hierarchy.Group):
    def __init__(self, store) -> None:
        zarr.hierarchy.Group.__init__(self, store)
        self.abs_path = Path(store.path).absolute()
        self.sim_name = self.abs_path.name.replace(self.abs_path.suffix, "")
        self.plot = Plot(self)
        self.calc = Calc(self)
        self.reload()

    def __repr__(self) -> str:
        return f"Llyr('{self.sim_name}')"

    def __str__(self) -> str:
        return f"Llyr('{self.sim_name}')"

    def reload(self):
        self._update_class_dict()

    def _update_class_dict(self):
        for k, v in self.attrs.items():
            self.__dict__[k] = v

    @property
    def pp(self):
        return self.tree(expand=True)

    @property
    def p(self):
        print(self.tree())

    @property
    def snap(self):
        self.plot.snapshot_png("stable")

    def c_to_comp(self, c):
        return ["mx", "my", "mz"][c]

    def get_mode(self, dset: str, f: float, c: int = None):
        if f"modes/{dset}/arr" not in self:
            print("Calculating modes ...")
            self.calc.modes(dset)
        fi = int((np.abs(self[f"modes/{dset}/freqs"][:] - f)).argmin())
        arr = self[f"modes/{dset}/arr"][fi]
        if c is None:
            return arr
        else:
            return arr[..., c]

    def check_path(self, dset: str, force: bool = False):
        if dset in self:
            if force:
                del self[dset]
            else:
                raise NameError(
                    f"The dataset:'{dset}' already exists, you can use 'force=True'"
                )

    def make_report(self, dset="m"):
        os.makedirs(f"{self.apath}/report")
        r = self.plot.report(dset=dset, save=f"{self.apath}/report/spectra.pdf")
        for peak in r.peaks:
            self.plot.anim(
                dset=dset,
                f=peak.freq,
                save_path=f"{self.apath}/report/{peak.freq:.2f}.gif",
            )


def iplot(path, label="", xstep=2, c="mx"):
    def plot_mode(m, ax, f):
        mode = m.get_mode("m", f, 2)[0]
        mode = np.tile(mode, (2, 2))
        extent = [0, mode.shape[1] * m.dx * 1e9, 0, mode.shape[0] * m.dy * 1e9]
        ax.imshow(
            np.angle(mode),
            aspect="equal",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            extent=extent,
            alpha=np.abs(mode) / np.abs(mode).max(),
        )

    paths = sorted(
        glob(f"{path}/*.zarr"),
        key=lambda x: int(x.split("/")[-1].replace(".zarr", "")),
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
    names = []
    arr = []
    for p in paths:
        m = op(p)
        names.append(m.name)
        x, y = m.calc.fft_tb(c, tmax=500, normalize=True)
        arr.append(y)
    arr = np.array(arr).T

    xlabels = np.array([int(p.split("/")[-1].replace(".zarr", "")) for p in paths])
    lstep = xlabels[1] - xlabels[0]
    ax1.imshow(
        arr,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[xlabels[0], xlabels[-1] + lstep, x.min(), x.max()],
        cmap="Reds",
    )
    ax1.set_ylim(6, 15)
    ax1.set_xticks(xlabels[::xstep] + lstep / 2)
    ax1.set_xticklabels(xlabels[::xstep])

    ax1.set_xlabel(label)
    ax1.grid(color="gray", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Frequency (GHz)")
    hline = ax1.axhline(x.max())
    vline = ax1.axvline(xlabels[0])

    def onclick(event):
        x = int(event.xdata)
        new_x = xlabels[0]
        while True:
            if new_x + lstep > x:
                break
            else:
                new_x += lstep
        x = new_x
        ax2.cla()
        ax2.set_title(f"{x} nm  -  {event.ydata:.2f} GHz")
        m = op(f"{path}/{x}.zarr")
        plot_mode(m, ax2, event.ydata)

        vline.set_data([new_x + lstep / 2, new_x + lstep / 2], [0, 1])
        hline.set_data([0, 1], [event.ydata, event.ydata])
        fig.tight_layout()

    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    fig.canvas.mpl_connect("button_press_event", onclick)
