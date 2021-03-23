

class Ovf:
    def save_ovf(self,dset:str,name:str,t:int=0) ->None :
        """Saves the given dataset to a valid OOMMF V2 ovf file"""
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