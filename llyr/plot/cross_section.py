import matplotlib.pyplot as plt

from ..base import Base


class cross_section(Base):
    def plot(self, dset="m", t=-1, z=0, y=330, c=2):
        arr = self.llyr[dset][t, z, y, :, c]
        plt.figure()
        plt.plot(arr)

        return self
