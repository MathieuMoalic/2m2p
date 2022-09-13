from glob import glob
from timeit import repeat
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Button
import numpy as np
import zarr
import peakutils

from ._utils import get_cmaps, make_cmap


def iplotp2(op, path, xstep=2, comps=None, fmin=0, fmax=20, unit="mT"):
    if comps is None:
        comps = [0, 2]
    paths = sorted(
        glob(f"{path}/*.zarr"), key=lambda x: int(x.split("/")[-1].replace(".zarr", ""))
    )  # [:110]
    xlabels = np.array([int(p.split("/")[-1].replace(".zarr", "")) for p in paths])
    lstep = xlabels[1] - xlabels[0]
    fig = plt.figure(figsize=(8, 4), dpi=150)
    gs = fig.add_gridspec(3, 6)
    ax1 = fig.add_subplot(gs[:, :3])
    ax1.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
    gs.update(left=0.08, right=0.99, top=0.88, bottom=0.01, wspace=0.1, hspace=0.1)
    cmaps, handles = get_cmaps()
    cm1 = make_cmap((139, 233, 253, 0), (139, 233, 253, 255), 256)
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
            norm=mpl.colors.LogNorm(vmin=0.01),
            extent=[
                xlabels[0] - lstep / 2,
                xlabels[-1] + lstep / 2,
                freqs.min(),
                freqs.max(),
            ],
            # cmap=cm1,
            cmap="cmo.amp",
        )
    ax1.legend(handles=[handles[i] for i in comps], fontsize=8)
    ax1.set_ylim(fmin, fmax)
    ax1.set_xticks(xlabels[::xstep])
    ax1.set_title(path)
    ax1.grid(color="gray", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("Frequency (GHz)")
    hline = ax1.axhline((freqs.max() - freqs.min()) / 2, ls="--", lw=0.8, c="k")
    vline = ax1.axvline(xlabels[0], ls="--", lw=0.8, c="k")
    axes = [
        fig.add_subplot(gs[i : i + 1, j : j + 1]) for i in [0, 1, 2] for j in [3, 4, 5]
    ]
    for ax in axes:
        ax.set(xticks=[], yticks=[])
    axes = np.array(axes).reshape(3, 3)

    class state:
        thres: float
        x = xlabels[0]
        y = 10
        m = op(f"{paths[0]}")
        peaks = []
        peak_index = 0

        def update_thres(self, thres):
            ax1.get_images()[0].set_norm(mpl.colors.LogNorm(vmin=thres))
            self.thres = thres
            ax1.set_title(f"vmin = {self.thres}")

    s = state()
    s.update_thres(0.002)

    def plot_mode(m, f):
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
            # if i in [0, 1]:
            #     arr = np.abs(mode[..., 1]) + np.abs(mode[..., 0])
            axes[2, i].imshow(
                np.abs(arr),
                cmap="inferno",
                vmin=0,
                vmax=absmax,
                interpolation="None",
                aspect="equal",
            )

    def pick_point(x: float, y: float, snap: bool):
        for ax in axes.flatten():
            ax.cla()
            ax.set(xticks=[], yticks=[])
        # ax1.set_title(f"{x=} - {y=}")
        x = xlabels[np.abs(xlabels - x).argmin()]
        m = op(f"{path}/{x:0>4}.zarr")
        if snap:
            fft = m.fft.m.max[2:, 0]
            freqs = m.fft.m.freqs[2:]  # * 1e-9
            peaks = freqs[peakutils.indexes(fft, thres=s.thres, min_dist=2)]
            s.peaks = peaks
            s.peak_index = np.abs(peaks - y).argmin()
            y = peaks[s.peak_index]
        vline.set_data([x, x], [0, 1])
        hline.set_data([0, 1], [y, y])
        plot_mode(m, y)
        axes[0, 1].text(
            0.5,
            1.3,
            f"{x} {unit} -  {y:.2f} GHz",
            va="center",
            ha="center",
            transform=axes[0, 1].transAxes,
        )
        s.x, s.y, s.m = x, y, m

    def onclick(event):
        if event.inaxes == ax1:
            if event.button.name == "RIGHT":
                pick_point(event.xdata, event.ydata, True)
            else:
                pick_point(event.xdata, event.ydata, False)

    def onpress(event):
        if event.key == "-":
            s.update_thres(s.thres * 1.1)
        if event.key == "=":
            s.update_thres(s.thres * 0.9)
        if event.key == "q":
            s.m.plot.anim(
                "m",
                f=s.y,
                save_path=f"figs/gifs4/{s.m.sim_name}_{s.y:.2f}.gif",
                repeat=1,
            )
        if event.key == "right":
            pick_point(s.x + lstep, s.y, True)
        if event.key == "left":
            pick_point(s.x - lstep, s.y, True)
        if event.key == "up":
            pick_point(s.x, s.peaks[s.peak_index + 1], True)
        if event.key == "down":
            pick_point(s.x, s.peaks[s.peak_index - 1], True)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onpress)

    return fig, ax1
