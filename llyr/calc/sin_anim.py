import numpy as np

from ..base import Base


class sin_anim(Base):
    def calc(self, dset, f, t=40):
        s = self.llyr.h5.shape(dset)
        arr_anim = np.zeros((t, s[2], s[3], 3))
        for c in range(3):
            arr = self.llyr.calc.mode(dset, f, c)
            times = np.linspace(0, 1 / f, t)
            y = np.zeros(
                (times.shape[0], arr.shape[1], arr.shape[0]), dtype=np.complex64
            )
            for i in range(arr.shape[1]):
                for j in range(arr.shape[0]):
                    phi = np.angle(arr[j, i])
                    r = np.abs(arr[j, i])
                    y[:, i, j] = (
                        r * np.cos(2 * np.pi * f * times + phi)
                        + r * np.sin(2 * np.pi * f * times + phi) * 1j
                    )
            amps = np.abs(y)
            amps /= amps.max()
            angle = np.angle(y)
            arr = angle * amps
            arr_anim[..., c] = arr
        return arr_anim
