import os
from pathlib import Path
from glob import glob
import shutil

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button
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

    def rm(self, dset: str):
        shutil.rmtree(f"{self.abs_path}/{dset}", ignore_errors=True)

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

    def comp_to_c(self, c):
        return {"mx": 0, "my": 1, "mz": 2}[c]

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
                self.rm(dset)
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


def get_cmaps():
    cmaps = []
    for a, b, c in zip((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(1, a, N)
        vals[:, 1] = np.linspace(1, b, N)
        vals[:, 2] = np.linspace(1, c, N)
        vals[:, 3] = np.linspace(0, 1, N)
        cmaps.append(mpl.colors.ListedColormap(vals))
    handles = [
        mpl.patches.Patch(color="red", label="mx"),
        mpl.patches.Patch(color="green", label="my"),
        mpl.patches.Patch(color="blue", label="mz"),
    ]
    return cmaps, handles


def iplot(path, xstep=2, comps=None, fmin=0, fmax=20):
    if comps is None:
        comps = [0, 2]
    paths = sorted(
        glob(f"{path}/*.zarr"), key=lambda x: int(x.split("/")[-1].replace(".zarr", ""))
    )
    xlabels = np.array([int(p.split("/")[-1].replace(".zarr", "")) for p in paths])
    lstep = xlabels[1] - xlabels[0]
    fig = plt.figure(figsize=(8, 5), dpi=150)
    gs = fig.add_gridspec(3, 6)
    ax1 = fig.add_subplot(gs[:, :3])
    gs.update(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    cmaps, handles = get_cmaps()
    for comp in comps:
        arr = []
        for p in paths:
            m = op(p)
            arr.append(m.fft.m.fft[2:, comp])
        arr = np.array(arr).T
        ts = m.m.attrs["t"]
        x = np.fft.rfftfreq(m.m.shape[0], (ts[-1] - ts[0]) / len(ts))[2:] * 1e-9
        ax1.imshow(
            arr,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            norm=mpl.colors.LogNorm(),
            extent=[xlabels[0] - lstep / 2, xlabels[-1] + lstep / 2, x.min(), x.max()],
            cmap=cmaps[comp],
        )
    ax1.legend(handles=[handles[i] for i in comps], fontsize=8)
    ax1.set_ylim(fmin, fmax)
    ax1.set_xticks(xlabels[::xstep])
    ax1.set_title(path)
    ax1.grid(color="gray", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Frequency (GHz)")
    hline = ax1.axhline((x.max() - x.min()) / 2)
    vline = ax1.axvline(xlabels[0])
    q = ""
    axes = [
        fig.add_subplot(gs[i : i + 1, j : j + 1]) for i in [0, 1, 2] for j in [3, 4, 5]
    ]
    for ax in axes:
        ax.set(xticks=[], yticks=[])
    axes = np.array(axes).reshape(3, 3)

    def plot_mode(m, f):
        for ax in axes.flatten():
            ax.cla()
            ax.set(xticks=[], yticks=[])
        for i in range(3):
            axes[0, i].text(
                0.5,
                1.1,
                ["mx", "my", "mz"][i],
                va="center",
                ha="center",
                transform=axes[0, i].transAxes,
            )
        mode = m.get_mode("m", f)[0]
        absmax = np.max(np.abs(mode))
        for i in range(3):
            arr = mode[..., i]
            axes[0, i].imshow(
                np.angle(arr),
                aspect="equal",
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="None",
                alpha=np.clip(np.abs(arr) / np.abs(arr).max(), 0, 1),
            )
            axes[1, i].imshow(
                np.angle(arr),
                aspect="equal",
                cmap="hsv",
                vmin=-np.pi,
                vmax=np.pi,
                interpolation="None",
            )
            axes[2, i].imshow(
                np.abs(arr),
                cmap="inferno",
                vmin=0,
                vmax=absmax,
                interpolation="None",
                aspect="equal",
            )
            axes[2, i].set_title(f"amp = {np.sum(np.abs(arr)):.2e}", fontsize=8)

    def onclick(event):
        x = xlabels[np.abs(xlabels - event.xdata).argmin()]
        plot_mode(op(f"{path}/{x:0>3}.zarr"), event.ydata)
        axes[0, 1].text(
            0.5,
            1.3,
            f"{q} - {x} nm -  {event.ydata:.2f} GHz",
            va="center",
            ha="center",
            transform=axes[0, 1].transAxes,
        )
        vline.set_data([x, x], [0, 1])
        hline.set_data([0, 1], [event.ydata, event.ydata])

    # def ch_color(_):
    #     q = "hi"

    # btn = Button(plt.axes([0.81, 0.000001, 0.1, 0.075]), "Peak Snap")
    # btn.on_clicked(ch_color)
    fig.canvas.mpl_connect("button_press_event", onclick)
