from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button
import numpy as np
import zarr
import peakutils

from ._utils import get_cmaps


def iplotp(op, path, xstep=2, comps=None, fmin=0, fmax=20):
    if comps is None:
        comps = [0, 2]
    paths = sorted(
        glob(f"{path}/*.zarr"), key=lambda x: int(x.split("/")[-1].replace(".zarr", ""))
    )
    xlabels = np.array([int(p.split("/")[-1].replace(".zarr", "")) for p in paths])
    lstep = xlabels[1] - xlabels[0]
    fig = plt.figure(figsize=(8, 5), dpi=150)
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[:])
    gs.update(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    cmaps, handles = get_cmaps()
    for comp in comps:
        arr = []
        for p in paths:
            m = op(p)
            arr.append(m.fft.m.max[2:, comp])
        arr = np.array(arr).T
        ts = m.m.attrs["t"]
        freqs = np.fft.rfftfreq(m.m.shape[0], (ts[-1] - ts[0]) / len(ts))[2:] * 1e-9
        ax1.imshow(
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
    ax1.legend(handles=[handles[i] for i in comps], fontsize=8)
    ax1.set_ylim(fmin, fmax)
    ax1.set_xticks(xlabels[::xstep])
    ax1.set_title(path)
    ax1.grid(color="gray", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Frequency (GHz)")
    return fig, ax1
