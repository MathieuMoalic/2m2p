from glob import glob
from tokenize import Name
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button
import numpy as np
import zarr
import peakutils

from ._utils import get_cmaps


def iplotp(op, path, xstep=2, comps=None, fmin=0, fmax=20, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 5), dpi=150)
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[:])
    else:
        fig = ax.figure

    if comps is None:
        comps = [0, 2]
    paths = sorted(
        glob(f"{path}/*.zarr"), key=lambda x: int(x.split("/")[-1].replace(".zarr", ""))
    )
    if len(paths) == 0:
        raise NameError("Wrong path")
    xlabels = np.array([int(p.split("/")[-1].replace(".zarr", "")) for p in paths])
    lstep = xlabels[1] - xlabels[0]
    cmaps, handles = get_cmaps()
    for comp in comps:
        arr = []
        for p in paths:
            m = op(p)
            arr.append(m.fft.m.max[2:, comp])
        arr = np.array(arr).T
        ts = m.m.attrs["t"]
        freqs = np.fft.rfftfreq(m.m.shape[0], (ts[-1] - ts[0]) / len(ts))[2:] * 1e-9
        ax.imshow(
            arr,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            norm=mpl.colors.LogNorm(),
            extent=[
                xlabels[0] - lstep / 2,
                xlabels[-1] + lstep / 2,
                freqs.min(),
                freqs.max(),
            ],
            cmap=cmaps[comp],
        )
    # ax.legend(handles=[handles[i] for i in comps], fontsize=8)
    ax.set_ylim(fmin, fmax)
    ax.set_xticks(xlabels[::xstep])
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Frequency (GHz)")
    return fig, ax
