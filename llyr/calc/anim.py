import numpy as np

from ..base import Base


class anim(Base):
    def calc(self, dset: str, f: float, t: int = 40, z: int = 0, periods: int = 1):
        mode = self.llyr.modes(dset, f)[z]
        tLi = np.linspace(0, 2 * np.pi * periods, t * periods)
        y = np.zeros(
            (tLi.shape[0], mode.shape[0], mode.shape[1], mode.shape[2]),
            dtype=np.float32,
        )
        for i, ti in enumerate(tLi):
            y[i] = np.real(mode * np.exp(1j * ti))
        return y / y.max()
