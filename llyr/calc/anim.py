import numpy as np

from ..base import Base


class anim(Base):
    def calc(self, dset: str, f: float, t: int = 40, periods: int = 1, norm=False):
        mode = self.m.get_mode(dset, f)
        # tLi = np.linspace(0, 2 * np.pi * periods, t * periods)
        # y = np.zeros(
        #     (tLi.shape[0], mode.shape[0], mode.shape[1], mode.shape[2], mode.shape[3]),
        #     dtype=np.float32,
        # )
        # for i, ti in enumerate(tLi):
        #     y[i] = np.real(mode * np.exp(1j * ti))
        arr = mode[None, ...] * np.exp(
            1j * np.linspace(0, 2 * np.pi, 40)[..., None, None, None]
        )
        arr /= arr.max()
        if norm:
            arr /= np.linalg.norm(arr, axis=-1)[..., None]
        return arr
