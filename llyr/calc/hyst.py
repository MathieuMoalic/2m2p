import numpy as np

from ..base import Base


class hyst(Base):
    def calc(self):
        self.m.rm(f"hyst/m")
        B = self.m.table.B_extz[:]
        m = np.average(
            np.ma.masked_equal(self.m.m[: len(B), :, :, :, 2], 0), axis=(1, 2, 3)
        )
        self.m.create_dataset(f"hyst/B", data=B, chunks=False)
        self.m.create_dataset(f"hyst/m", data=m, chunks=False)
