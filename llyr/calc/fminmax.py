import numpy as np
from ..base import Base


class fminmax(Base):
    def calc(self, freqs, spec, fmin, fmax, normalize=False):
        fimin = np.abs(freqs - fmin).argmin()
        fimax = np.abs(freqs - fmax).argmin()
        freqs, specs = freqs[fimin:fimax], spec[fimin:fimax]
        if normalize:
            specs /= specs.max()
        return freqs, specs
