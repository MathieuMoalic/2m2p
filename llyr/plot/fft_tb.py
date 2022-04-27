import matplotlib.pyplot as plt

from ..base import Base


class fft_tb(Base):
    def plot(
        self,
        fmin=5,
        fmax=25,
        fft_tmin=0,
        fft_tmax=-1,
        fft_tstep=1,
        thres=0.01,
        min_dist=2,
        axes=None,
    ):
        if axes is None:
            self.fig, self.axes = plt.subplots(1, 3, sharex=True, figsize=(7, 3))
        else:
            self.fig = axes[0].figure
            self.axes = axes
        # for dset, ax in zip(["mx", "my", "mz"], self.axes):
        # freqs, spec = self.llyr.calc.fft_tb(
        #     dset, tmax=fft_tmax, tmin=fft_tmin, tstep=fft_tstep
        # )
        freqs = self.llyr.fft.m.freqs[:]
        spec = self.llyr.fft.m.max[:]
        freqs, spec = self.llyr.calc.fminmax(freqs, fmin, fmax, spec=spec)
        for comp in range(3):
            ax = axes[comp]
            ax.plot(freqs, spec[:, comp])
            peaks = self.llyr.calc.peaks(
                freqs, spec[:, comp], thres=thres, min_dist=min_dist
            )
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
                0.5,
                0.9,
                ["mx", "my", "mz"][comp],
                transform=ax.transAxes,
                fontweight="bold",
                ha="center",
                va="center",
                fontsize=16,
            )
        # self.axes[0].text(
        #     0,
        #     1.1,
        #     self.llyr.sim_name,
        #     transform=self.axes[0].transAxes,
        #     fontweight="bold",
        #     ha="left",
        #     va="center",
        #     fontsize=12,
        #     bbox=dict(
        #         boxstyle="square",
        #         ec=(1.0, 0.8, 0.8),
        #         fc=(1.0, 0.8, 0.8),
        #     ),
        # )
        self.fig.tight_layout()
        return self
