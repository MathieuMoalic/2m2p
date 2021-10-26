import matplotlib.pyplot as plt
import peakutils

from ..base import Base


class fft_tb(Base):
    def plot(
        self,
        tmin=0,
        tmax=-1,
        fft_tmin=0,
        fft_tmax=-1,
        tstep=1,
        thres=0.15,
        axes=None,
    ):
        if axes is None:
            _, axes = plt.subplots(3, 1, sharex=True)
        comps = ["mx", "my", "mz"]
        fss = []
        for i in range(3):
            c, ax = comps[i], axes[i]
            x, y = self.llyr.calc.fft_tb(c, tmax=fft_tmax, tmin=fft_tmin, tstep=tstep)
            ax.plot(x[tmin:tmax], y[tmin:tmax])
            list_peaks = peakutils.indexes(y, thres=thres, min_dist=4)
            fs = [x[i] for i in list_peaks]
            fss.append(fs)
            for f in fs:
                ax.axvline(f, ls="--", c="gray")
                ax.text(
                    f,
                    y[tmin:].max() * 1.15,
                    f"{f:.2f}",
                    fontsize=5,
                    rotation=45,
                    ha="left",
                    va="center",
                )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlim(5, 25)
        return self
