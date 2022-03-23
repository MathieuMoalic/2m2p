import numpy as np
from ..base import Base


class fminmax(Base):
    def calc(self, freqs, fmin, fmax, spec=None, normalize=False):
        fimin = np.abs(freqs - fmin).argmin()
        fimax = np.abs(freqs - fmax).argmin()
        freqs = freqs[fimin:fimax]
        if spec is not None:
            spec = spec[fimin:fimax]
            if normalize:
                spec /= spec.max()
            return freqs, spec
        return freqs
