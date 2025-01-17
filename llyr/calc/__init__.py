from .disp import disp
from .fft_tb import fft_tb
from .fft import fft
from .modes import modes
from .hyst import hyst
from .bad_modes import bad_modes
from .sk_number import sk_number
from .peaks import peaks
from .fminmax import fminmax
from .anim import anim


class Calc:
    def __init__(self, llyr):
        self.disp = disp(llyr).calc
        self.disp_da = disp(llyr).calc_da
        self.fft_tb = fft_tb(llyr).calc
        self.fft = fft(llyr).calc
        self.modes = modes(llyr).calc
        self.hyst = hyst(llyr).calc
        self.bad_modes = bad_modes(llyr).calc
        self.sk_number = sk_number(llyr).calc
        self.peaks = peaks(llyr).calc
        self.npeaks = peaks(llyr).npeaks
        self.fminmax = fminmax(llyr).calc
        self.anim = anim(llyr).calc
