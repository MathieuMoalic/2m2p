from collections import namedtuple

import peakutils

from ..base import Base


class peaks(Base):
    def calc(self, x, y, thres=0.01, min_dist=2):
        Peak = namedtuple("Peak", "idx freq amp")
        idx = peakutils.indexes(y, thres=thres, min_dist=min_dist)
        peak_amp = [y[i] for i in idx]
        freqs = [float(f"{x[i]:.2f}") for i in idx]
        return [Peak(i, f, a) for i, f, a in zip(idx, freqs, peak_amp)]

    def npeaks(self, x, y, peaknb, min_dist=2):
        Peak = namedtuple("Peak", "idx freq amp")
        thres = 0
        while len(idx) < peaknb:
            idx = peakutils.indexes(y, thres=thres, min_dist=min_dist)
            thres += 0.01
        peak_amp = [y[i] for i in idx]
        freqs = [float(f"{x[i]:.2f}") for i in idx]
        return [Peak(i, f, a) for i, f, a in zip(idx, freqs, peak_amp)]
