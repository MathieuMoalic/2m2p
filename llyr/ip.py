from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import peakutils

from ._utils import make_cmap


def ipp(op, path, xstep=2, comp=0, fmin=0, fmax=20, title="nm", anim=False):
    paths = sorted(
        glob(f"{path}/*.zarr"), key=lambda x: int(x.split("/")[-1].replace(".zarr", ""))
    )
    xlabels = np.array([int(p.split("/")[-1].replace(".zarr", "")) for p in paths])
    lstep = xlabels[1] - xlabels[0]
    fig = plt.figure(figsize=(8, 4), dpi=200)
    gs = fig.add_gridspec(1, 2)
    ax_plot = fig.add_subplot(gs[:, 0])
    ax_plot.tick_params(
        axis="x", bottom=False, top=True, labelbottom=False, labeltop=True
    )
    gs.update(left=0.08, right=0.99, top=0.88, bottom=0.03, wspace=0.1, hspace=0.1)
    cm1 = make_cmap((40, 42, 54, 0), (139, 233, 253, 255), 256)
    arr = []
    for p in paths:
        m = op(p)
        arr.append(m.fft.m.max[2:, comp])
    arr = np.array(arr).T
    ts = m.m.attrs["t"]
    freqs = np.fft.rfftfreq(m.m.shape[0], (ts[-1] - ts[0]) / len(ts))[2:] * 1e-9
    ax_plot.imshow(
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
        cmap=cm1,
    )
    ax_plot.set_ylim(fmin, fmax)
    ax_plot.set_xticks(xlabels[::xstep])
    ax_plot.set_title(title)
    ax_plot.grid(color="gray", linestyle="--", linewidth=0.5)
    ax_plot.set_ylabel("Frequency (GHz)")
    hline = ax_plot.axhline((freqs.max() - freqs.min()) / 2, ls="--", lw=0.8, c="#ffb86c")
    vline = ax_plot.axvline(xlabels[0], ls="--", lw=0.8, c="#ffb86c")

    class state:
        thres: float
        x = xlabels[0]
        y = 10.0
        m = op(f"{paths[0]}")
        peaks = []
        peak_index = 0

        def update_thres(self, thres):
            ax_plot.get_images()[0].set_norm(mpl.colors.LogNorm(vmin=thres))
            self.thres = thres

    s = state()
    s.update_thres(0.002)
    ax_mode = fig.add_subplot(gs[:, 1])

    def plot_mode(m, f):
        # arr = m.get_mode("m", f)[0, ..., comp]
        arrs = m.calc.anim("m", f, periods=1)[:, 0, :, :, comp]
        arr = arrs[0]
        ax_mode.imshow(
            np.angle(arr),
            aspect="equal",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            interpolation="None",
            alpha=np.clip(np.abs(arr) / np.abs(arr).max(), 0, 1),
        )
        if anim:

            def run(t):
                ax_mode.get_images()[0].set_data(np.angle(arrs[t]))

            ani = mpl.animation.FuncAnimation(
                fig, run, interval=50, frames=np.arange(1, arrs.shape[0], dtype="int")
            )

    def pick_point(x: float, y: float, snap: bool):
        ax_mode.cla()
        ax_mode.set(xticks=[], yticks=[])
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
        ax_mode.set_title(f"diameter = {x} nm   f={y:.2f} GHz")
        plot_mode(m, y)
        s.x, s.y, s.m = x, y, m

    def onclick(event):
        if event.inaxes == ax_plot:
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

    return fig, ax_plot
