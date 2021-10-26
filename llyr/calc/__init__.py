from .disp import disp
from .fft_tb import fft_tb
from .fft import fft
from .mode import mode
from .sk_number import sk_number


class Calc:
    def __init__(self, llyr):
        self.disp = disp(llyr).calc
        self.fft_tb = fft_tb(llyr).calc
        self.fft = fft(llyr).calc
        self.mode = mode(llyr).calc
        self.sk_number = sk_number(llyr).calc
