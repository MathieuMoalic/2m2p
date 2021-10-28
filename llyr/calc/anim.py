import numpy as np

from ..base import Base


class anim(Base):
    def calc(self, dset, f, t=40):
        mode = self.llyr.modes(dset, f)
        tLi = np.linspace(0, 2 * np.pi, t)
        y = np.zeros(
            (tLi.shape[0], mode.shape[0], mode.shape[1], mode.shape[2]),
            dtype=np.float32,
        )
        for i, ti in enumerate(tLi):
            y[i] = np.real(mode * np.exp(1j * ti))
        return y
