import numpy as np

from ..base import Base


class sk_number(Base):
    def calc(self, dset: str, z: int = 0, t: int = 0):
        spin_grid = self.llyr[dset][t, z, :, :, :]
        spin_pad = np.pad(
            spin_grid, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0.0
        )
        sk_nb = (
            np.cross(
                spin_pad[2:, 1:-1, :],  # s(i+1,j)
                spin_pad[1:-1, 2:, :],  # s(i,j+1)
                axis=2,
            )
            + np.cross(
                spin_pad[:-2, 1:-1, :],  # s(i-1,j)
                spin_pad[1:-1, :-2, :],  # s(i,j-1)
                axis=2,
            )
            - np.cross(
                spin_pad[:-2, 1:-1, :],  # s(i-1,j)
                spin_pad[1:-1, 2:, :],  # s(i,j+1)
                axis=2,
            )
            - np.cross(
                spin_pad[2:, 1:-1, :],  # s(i+1,j)
                spin_pad[1:-1, :-2, :],  # s(i,j-1)
                axis=2,
            )
        )
        sk_nb = -1 * np.einsum("ijk,ijk->ij", spin_grid, sk_nb) / (16 * np.pi)
        return np.sum(sk_nb.flatten())
