from .anim import anim
from .anim2 import anim2
from .fft_tb import fft_tb
from .imshow import imshow
from .modes import modes
from .snapshot import snapshot
from .snapshot_png import snapshot_png
from .report import report
from .sin_anim import sin_anim
from .cross_section import cross_section
from .p3d import p3d


class Plot:
    def __init__(self, llyr):
        self.anim = anim(llyr).plot
        self.anim2 = anim2(llyr).plot
        self.fft_tb = fft_tb(llyr).plot
        self.imshow = imshow(llyr).plot
        self.modes = modes(llyr).plot
        self.mode = modes(llyr).plot_one
        self.mode_v2 = modes(llyr).plot_one_v2
        self.snapshot = snapshot(llyr).plot
        self.snapshot_png = snapshot_png(llyr).plot
        self.report = report(llyr).plot
        self.sin_anim = sin_anim(llyr).plot
        self.cross_section = cross_section(llyr).plot
        self.p3d = p3d(llyr).plot
