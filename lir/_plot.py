import matplotlib.pyplot as plt
import cmocean
import numpy as np
import h5py

class Plot:

    def dplot(self,dset:str='disp',fmin:int=25,fmax:int=450,kwidth:int=150,dpi:int=150):
        fig,ax = plt.subplots(1,1,dpi=dpi)
        kmid = self.shape(dset)[1]//2
        kmin = kmid - kwidth
        kmax = kmid + kwidth
        arr = self[dset][fmin:fmax,kmin:kmax]
        kvecs = self.kvecs[kmin:kmax]
        freqs = self.freqs[fmin:fmax]
        ax.imshow(arr,origin='lower',aspect='auto',cmap='cmo.amp',extent=[kvecs.min(), kvecs.max(), freqs.min(), freqs.max()])
        ax.set_ylabel("Frequency (GHz)")
        ax.set_ylabel(r"Wavevectors (nm$^{-1}$)")
        fig.colorbar(ax.get_images()[0],ax=ax)
        return ax

