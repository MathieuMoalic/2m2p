import matplotlib.pyplot as plt

from ..base import Base


class fft_tb(Base):
    def plot(
        self,
        fmin=5,
        fmax=25,
        fft_tmin=0,
        fft_tmax=-1,
        tstep=1,
        thres=0.01,
        axes=None,
    ):
        if axes is None:
            self.fig, self.axes = plt.subplots(1, 3, sharex=True, figsize=(7, 3))
        names = [r"$m_x$", r"$m_y$", r"$m_z$"]
        for i, ax in enumerate(self.axes):
            freqs, spec = self.llyr.calc.fft_tb(
                "m", i, tmax=fft_tmax, tmin=fft_tmin, tstep=tstep
            )
            freqs, spec = self.llyr.calc.fminmax(freqs, fmin, fmax, spec=spec)
            ax.plot(freqs, spec)
            peaks = self.llyr.calc.peaks(freqs, spec, thres=thres)
            for peak in peaks:
                ax.text(
                    peak.freq,
                    peak.amp + 0.03 * spec.max(),
                    f"{peak.freq:.2f}",
                    # fontsize=5,
                    rotation=90,
                    ha="center",
                    va="bottom",
                )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.9,
                0.9,
                names[i],
                transform=ax.transAxes,
                fontweight="bold",
                ha="center",
                va="center",
                fontsize=16,
            )
        self.axes[0].text(
            0,
            1.1,
            self.llyr.aname,
            transform=self.axes[0].transAxes,
            fontweight="bold",
            ha="left",
            va="center",
            fontsize=12,
            bbox=dict(
                boxstyle="square",
                ec=(1.0, 0.8, 0.8),
                fc=(1.0, 0.8, 0.8),
            ),
        )
        self.fig.tight_layout()
        return self
