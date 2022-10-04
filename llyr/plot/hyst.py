import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RadioButtons

from ..base import Base


class hyst(Base):
    def plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.subplots_adjust(bottom=0.16, top=0.94, right=0.99, left=0.08)
        b_ext = self.m.hyst.B[:]
        m_avr = self.m.hyst.m[:]
        passes = ["from 1T to -1T", "from -1T to 1T"]
        ax1.plot(b_ext[: len(b_ext) // 2], m_avr[: len(b_ext) // 2], label=passes[0])
        ax1.plot(
            b_ext[len(b_ext) // 2 :], m_avr[len(b_ext) // 2 :], label=passes[1], ls="--"
        )
        ax1.legend()
        B_sel = 10
        vline = ax1.axvline(b_ext[B_sel], c="gray", ls=":")

        def onclick(event):
            if event.inaxes == ax1:
                B_sel = np.abs(b_ext[: len(b_ext) // 2] - event.xdata).argmin()
                ax2.cla()
                if np.abs(m_avr[B_sel] - event.ydata) < np.abs(
                    m_avr[len(m_avr) // 2 :][::-1][B_sel] - event.ydata
                ):
                    self.m.plot.snapshot("m", t=B_sel, ax=ax2)
                    ax2.set_title(f"B_ext = {b_ext[B_sel]:.3f} T; from 1T to -1T")
                else:
                    self.m.plot.snapshot("m", t=len(b_ext) - B_sel, ax=ax2)
                    ax2.set_title(f"B_ext = {b_ext[B_sel]:.3f} T; from -1T to 1T")

                vline.set_data([b_ext[B_sel], b_ext[B_sel]], [0, 1])
                fig.canvas.draw()

        fig.canvas.mpl_connect("button_press_event", onclick)
