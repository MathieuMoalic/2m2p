import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from ..base import Base


class sin_anim(Base):
    def plot(self, dset, f):
        fig, axes = plt.subplots(1, 3, figsize=(7, 2), dpi=200)
        arr = self.llyr.calc.sin_anim(dset, f)
        axes[0].imshow(arr[0, :, :, 0])
        axes[1].imshow(arr[0, :, :, 1])
        axes[2].imshow(arr[0, :, :, 2])

        def run(t):
            axes[0].get_images()[0].set_data(arr[t, :, :, 0])
            axes[1].get_images()[0].set_data(arr[t, :, :, 1])
            axes[2].get_images()[0].set_data(arr[t, :, :, 2])
            return axes

        ani = mpl.animation.FuncAnimation(
            fig, run, interval=1, frames=np.arange(1, arr.shape[0], dtype="int")
        )
        ani.save(
            f"{self.llyr.aname}_{dset}_{f}.mp4",
            writer="ffmpeg",
            fps=25,
            dpi=200,
            extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"],
        )
