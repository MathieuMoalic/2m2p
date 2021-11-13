from .anim import anim
from .fft_tb import fft_tb
from .imshow import imshow
from .modes import modes
from .snapshot import snapshot
from .snapshot_png import snapshot_png
from .report import report
from .sin_anim import sin_anim
from .cross_section import cross_section


class Plot:
    def __init__(self, llyr):
        self.anim = anim(llyr).plot
        self.fft_tb = fft_tb(llyr).plot
        self.imshow = imshow(llyr).plot
        self.modes = modes(llyr).plot
        self.snapshot = snapshot(llyr).plot
        self.snapshot_png = snapshot_png(llyr).plot
        self.report = report(llyr).plot
        self.sin_anim = sin_anim(llyr).plot
        self.cross_section = cross_section(llyr).plot
