import h5py  # type: ignore
import numpy as np
from typing import *
import struct

from lir._make import Make
from lir._transform import Transform


class Lir(Make, Transform):
    def __init__(self, h5_path: str, loadpath: Optional[str] = None, force=False):
        self.force = force
        self.h5_path = h5_path
        self.name = h5_path.split("/")[-1][:-3]
        if loadpath is not None:
            self.make(loadpath)
        self.getitem_dset: Optional[str] = None

    def __getitem__(
        self,
        index: Union[str, Tuple[Union[int, slice], ...]],
    ) -> Union["H5", float, np.ndarray]:
        with h5py.File(self.h5_path, "r") as f:

            # if slicing
            if isinstance(index, (slice, tuple, int)):
                # if dset is defined
                if isinstance(self.getitem_dset, str):
                    out_dset: np.ndarray = f[self.getitem_dset][index]
                    self.getitem_dset = None
                    return out_dset
                else:
                    raise AttributeError("You can only slice datasets")

            elif isinstance(index, str):
                # if dataset
                if index in list(f.keys()):
                    self.getitem_dset = index
                    return self
                # if attribute
                elif index in list(f.attrs.keys()):
                    out_attribute: float = f.attrs[index]
                    return out_attribute
                else:
                    raise KeyError("No such Dataset or Attribute")
            else:
                raise TypeError()

    def dset(self, name: str, data: np.ndarray) -> None:
        with h5py.File(self.h5_path, "a") as f:
            f.create_dataset(name, data=data)

    def a(
        self,
        name: str,
        key: str,
        val: Union[str, int, float, slice, Tuple[Union[int, slice], ...]],
    ) -> None:
        with h5py.File(self.h5_path, "a") as f:
            f[name].attrs[key] = val

    def t(self,script_name,ovf_folder="/mnt/g/Mathieu/simulations/stable",dset="stable",t=0):
        linux_ovf_name = f"{ovf_folder}/{self.name}.ovf"
        windows_ovf_name = f"G:{ovf_folder[6:]}/{self.name}.ovf"
        script_lines = self["script"].split("\n")
        load_line = f'm.loadfile("{windows_ovf_name}")\n'

        with open(script_name,"w") as f:
            prefix = ""
            for i,line in enumerate(script_lines):
                f.write(prefix+line+"\n")
                if "dind.setregio" in line:
                    f.write("\n")
                    f.write(load_line)
                    prefix = "// "

        self.save_ovf(dset,linux_ovf_name,t=t)

    def script(self,savepath=None):
        if savepath is None:
            print(self["script"])
        else:
            with open(savepath,"w") as f:
                f.writelines(self["script"])

    def save_ovf(self,dset,name,t=0):

        def whd(s):
            s += "\n"
            f.write(s.encode("ASCII"))
            
        arr = self[dset][t]
        out = arr.astype('<f4')
        out = out.tobytes()
        title = dset
        xstepsize, ystepsize, zstepsize = self["dx"], self["dy"], self["dz"], 
        xnodes, ynodes, znodes = arr.shape[2], arr.shape[1], arr.shape[0]
        xmin, ymin, zmin = 0, 0, 0
        xmax, ymax, zmax = xnodes*xstepsize, ynodes*ystepsize, znodes*zstepsize
        xbase, ybase, zbase = xstepsize/2, ystepsize/2, zstepsize/2
        valuedim = arr.shape[-1]
        valuelabels = "x y z"
        valueunits = "1 1 1"
        total_sim_time = "0"
        with open(name,"wb") as f:
            whd(f"# OOMMF OVF 2.0")
            whd(f"# Segment count: 1")
            whd(f"# Begin: Segment")
            whd(f"# Begin: Header")
            whd(f"# Title: {title}")
            whd(f"# meshtype: rectangular")
            whd(f"# meshunit: m")
            whd(f"# xmin: {xmin}")
            whd(f"# ymin: {ymin}")
            whd(f"# zmin: {zmin}")
            whd(f"# xmax: {xmax}")
            whd(f"# ymax: {ymax}")
            whd(f"# zmax: {zmax}")
            whd(f"# valuedim: {valuedim}")
            whd(f"# valuelabels: {valuelabels}")
            whd(f"# valueunits: {valueunits}")
            whd(f"# Desc: Total simulation time:  {total_sim_time}  s")
            whd(f"# xbase: {xbase}")
            whd(f"# ybase: {ybase}")
            whd(f"# zbase: {ybase}")
            whd(f"# xnodes: {xnodes}")
            whd(f"# ynodes: {ynodes}")
            whd(f"# znodes: {znodes}")
            whd(f"# xstepsize: {xstepsize}")
            whd(f"# ystepsize: {ystepsize}")
            whd(f"# zstepsize: {zstepsize}")
            whd(f"# End: Header")
            whd(f"# Begin: Data Binary 4")
            f.write(struct.pack("<f",1234567.0))
            f.write(out)
            whd(f"# End: Data Binary 4")
            whd(f"# End: Segment")
        

    @property
    def p(self) -> None:
        with h5py.File(self.h5_path, "r") as f:
            print("Datasets:")
            for key in list(f.keys()):
                print(f"    {key:<15}: {f[key].shape}")
            print("Attributes:")
            for key in list(f.attrs.keys()):
                if key != "script":
                    print(f"    {key:<15}= {f.attrs[key]}")

    def freqs(self) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as f:
            dt: float = f.attrs["dt"]
            tshape: int = f["WG"].shape[0]
            out: np.ndarray = np.fft.rfftfreq(tshape, dt * 1e9)
        return out

    def kvecs(self) -> np.ndarray:
        with h5py.File(self.h5_path, "r") as f:
            dx: float = f.attrs["dx"]
            xshape: int = f["WG"].shape[3]
            out: np.ndarray = (
                np.fft.fftshift(np.fft.fftfreq(xshape, dx) * 2 * np.pi) / 1e6
            )
        return out

    def d(self, dset: str) -> None:
        with h5py.File(self.h5_path, "a") as f:
            del f[dset]

    def move(self, source: str, destination: str) -> None:
        with h5py.File(self.h5_path, "a") as f:
            f.move(source, destination)
