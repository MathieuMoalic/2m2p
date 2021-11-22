from collections import namedtuple

import matplotlib.pyplot as plt
import peakutils
import numpy as np

from ..base import Base


class report(Base):
    def plot(
        self,
        dset="m",
        thres=0.01,
        min_dist=5,
        tmin=None,
        tmax=None,
        tstep=1,
        nb_modes=6,
        save=False,
    ):
        def get_spectra() -> dict:
            spectra = {}
            for c in range(3):
                y = self.llyr[f"table/{dset}"][slice(tmin, tmax, tstep), c]
                ts = self.llyr["table/t"][slice(tmin, tmax, tstep)]
                table_dt = (ts[-1] - ts[0]) / len(ts)
                x = np.fft.rfftfreq(y.shape[0], table_dt * tstep) * 1e-9
                y -= y[0]
                y -= np.average(y)
                y = np.multiply(y, np.hanning(y.shape[0]))
                y = np.fft.rfft(y)
                y = np.abs(y)
                spectra["freqs"] = x
                spectra[c] = y
            return spectra

        def get_peaks(s):
            Peak = namedtuple("Peak", "idx freq amp")
            all_peaks = {}
            for c in range(3):
                idx = peakutils.indexes(s[c], thres=thres, min_dist=min_dist)
                peak_amp = [s[c][i] for i in idx]
                freqs = [s["freqs"][i] for i in idx]
                all_peaks[c] = [Peak(i, f, a) for i, f, a in zip(idx, freqs, peak_amp)]
            no_dup_peaks = []
            all_freqs = []
            for peak_list in all_peaks.values():
                for peak in peak_list:
                    if peak.freq not in all_freqs:  # removes duplicate freqs
                        no_dup_peaks.append(peak)
                        all_freqs.append(peak.freq)
            sorted_peaks = sorted(no_dup_peaks, key=lambda x: x[2], reverse=True)[
                :nb_modes
            ]
            return sorted_peaks, all_peaks

        def get_modes(peaks):
            z = 0
            modes = []
            Mode = namedtuple("Mode", "idx freq amp mx my mz")
            ModeComp = namedtuple("ModeArr", "abs ang alpha")
            for peak in peaks:
                modes_comps = []
                arrs = self.llyr.modes(dset, peak.freq)[z, :, :]
                for comp in [0, 1, 2]:
                    arr = arrs[..., comp]
                    arr_abs = np.abs(arr)
                    arr_ang = np.angle(arr)
                    amax = arr_abs.max()
                    if amax == 0:
                        arr_alpha = arr_abs
                    else:
                        arr_alpha = arr_abs / amax
                    modes_comps.append(ModeComp(arr_abs, arr_ang, arr_alpha))

                modes.append(
                    Mode(
                        peak.idx,
                        peak.freq,
                        peak.amp,
                        modes_comps[0],
                        modes_comps[1],
                        modes_comps[2],
                    )
                )
            return modes

        def plot_spectra(gs_main, spectra, all_peaks):
            gs_spectra = gs_main[0].subgridspec(1, 3, wspace=0, hspace=0)
            axes = gs_spectra.subplots()
            axes[1].set_title(self.llyr.aname)
            for c, ax in enumerate(axes):
                peaks = all_peaks[c]
                ax.plot(spectra["freqs"], spectra[c])
                for peak in peaks:
                    ax.text(
                        peak.freq,
                        peak.amp + 0.03 * max(spectra[c]),
                        f"{peak.freq:.2f}",
                        fontsize=5,
                        rotation=90,
                        ha="center",
                        va="bottom",
                    )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

        def plot_modes(gs_main, modes):
            nb_rows = int(np.ceil(len(modes) / 3))
            gs_modes = gs_main[1:].subgridspec(nb_rows, 3, wspace=0, hspace=0)
            extent = [
                0,
                modes[0].my.abs.shape[1] * self.llyr.dx * 1e9,
                0,
                modes[0].mx.abs.shape[0] * self.llyr.dy * 1e9,
            ]
            # mode_list_max = np.max([np.max(mode.amp) for mode in modes])
            for i, mode in enumerate(modes):
                gs_mode = gs_modes[i].subgridspec(3, 3, wspace=0, hspace=0)
                axes = gs_mode.subplots()
                axes[0, 1].set_title(
                    f"{mode.freq:.2f} GHz (A={mode.amp:.1e})", fontsize=9
                )
                for c in range(3):
                    axes[0, c].imshow(
                        mode[c + 3].abs,
                        cmap="inferno",
                        # vmin=0,
                        # vmax=mode_list_max,
                        extent=extent,
                        interpolation="None",
                        aspect="equal",
                    )
                    axes[1, c].imshow(
                        mode[c + 3].ang,
                        aspect="equal",
                        cmap="hsv",
                        vmin=-np.pi,
                        vmax=np.pi,
                        interpolation="None",
                        extent=extent,
                    )
                    axes[2, c].imshow(
                        mode[c + 3].ang,
                        aspect="equal",
                        alpha=mode[c + 3].alpha,
                        cmap="hsv",
                        vmin=-np.pi,
                        vmax=np.pi,
                        interpolation="nearest",
                        extent=extent,
                    )
                for ax in axes.flatten():
                    ax.set(xticks=[], yticks=[])

        spectra = get_spectra()
        sorted_peaks, all_peaks = get_peaks(spectra)
        modes = get_modes(sorted_peaks)

        fig = plt.figure(constrained_layout=True, figsize=(7, 5))
        gs_main = fig.add_gridspec(4, 1)
        plot_spectra(gs_main, spectra, all_peaks)
        plot_modes(gs_main, modes)

        if isinstance(save, str):
            fig.savefig(save, dpi=100)
        elif isinstance(save, bool) and save:
            fig.savefig(f"{self.llyr.apath}/spectra.pdf", dpi=100)

        self.fig = fig
        self.peaks = sorted_peaks
        return self
