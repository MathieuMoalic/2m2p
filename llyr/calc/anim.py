import numpy as np

from ..base import Base


class anim(Base):
    def calc(self, dset: str, f: float, t: int = 40, periods: int = 1, norm=False):
        mode = self.m.get_mode(dset, f)
        tLi = np.linspace(0, 2 * np.pi * periods, t * periods)
        y = np.zeros(
            (tLi.shape[0], mode.shape[0], mode.shape[1], mode.shape[2], mode.shape[3]),
            dtype=np.float32,
        )
        for i, ti in enumerate(tLi):
            y[i] = np.real(mode * np.exp(1j * ti))
        y /= y.max()
        if norm:
            y /= np.linalg.norm(y, axis=-1)[..., None]
        return y


# class anim(Base):
#     def calc(
#         self, dset: str, f: float, t: int = 40, z: slice = slice(0, 1), periods: int = 1
#     ):
#         mode = self.m.get_mode(dset, f)[z][None, ...]
#         times = np.linspace(0, 2 * np.pi * periods, t * periods)
#         y = np.abs((times * mode.T).T)
#         return y / y.max()
