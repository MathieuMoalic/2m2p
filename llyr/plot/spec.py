from collections import namedtuple

import matplotlib.pyplot as plt
import peakutils
import numpy as np

from ..base import Base


class spec(Base):
    def plot(
        self,
        dset="m",
        thres=0.01,
        min_dist=5,
        c=0,
    ):
        def get_peaks(x, y):
            Peak = namedtuple("Peak", "idx freq amp")
            idx = peakutils.indexes(y, thres=thres, min_dist=min_dist)
            peak_amp = [y[i] for i in idx]
            freqs = [x[i] for i in idx]
            return [Peak(i, f, a) for i, f, a in zip(idx, freqs, peak_amp)]

        def plot_spectra(ax, x, y, peaks):
            ax.plot(x, y)
            for _, freq, amp in peaks:
                ax.text(
                    freq,
                    amp + 0.03 * max(y),
                    f"{freq:.2f}",
                    # fontsize=5,
                    rotation=90,
                    ha="center",
                    va="bottom",
                )
            ax.set_title(self.m.sim_name)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        def plot_modes(axes, f):
            for ax in axes.flatten():
                ax.cla()
                ax.set(xticks=[], yticks=[])
            mode = self.m.get_mode("m", f)[0]
            extent = [
                0,
                mode.shape[1] * self.m.dx * 1e9,
                0,
                mode.shape[0] * self.m.dy * 1e9,
            ]
            for c in range(3):
                abs_arr = np.abs(mode[:, :, c])
                phase_arr = np.angle(mode[:, :, c])
                axes[0, c].imshow(
                    abs_arr,
                    cmap="inferno",
                    # vmin=0,
                    # vmax=mode_list_max,
                    extent=extent,
                    interpolation="None",
                    aspect="equal",
                )
                axes[1, c].imshow(
                    phase_arr,
                    aspect="equal",
                    cmap="hsv",
                    vmin=-np.pi,
                    vmax=np.pi,
                    interpolation="None",
                    extent=extent,
                )
                axes[2, c].imshow(
                    phase_arr,
                    aspect="equal",
                    alpha=abs_arr / abs_arr.max(),
                    cmap="hsv",
                    vmin=-np.pi,
                    vmax=np.pi,
                    interpolation="nearest",
                    extent=extent,
                )

        fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        gs = fig.add_gridspec(1, 2)
        ax_spec = fig.add_subplot(gs[0, 0])
        x = self.m.fft.m.freqs[:]
        y = self.m.fft.m.max[:, c]
        peaks = get_peaks(x, y)
        plot_spectra(ax_spec, x, y, peaks)
        axes_modes = gs[0, 1].subgridspec(3, 3).subplots()
        vline = ax_spec.axvline(10, ls="--", lw=0.8, c="#ffb86c")
        # plot_modes(axes_modes, peaks[0].freq)

        def onclick(event):
            if event.inaxes == ax_spec:
                f = 10
                if event.button.name == "RIGHT":
                    freqs = [p.freq for p in peaks]
                    f = freqs[(np.abs(freqs - event.xdata)).argmin()]
                else:
                    f = event.xdata
                vline.set_data([f, f], [0, 1])
                plot_modes(axes_modes, f)
                fig.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)
        return fig, ax_spec
