import struct

import numpy as np


def save_ovf(path: str, arr: np.ndarray, dx: float, dy: float, dz: float) -> None:
    """Saves the given dataset for a given t to a valid OOMMF V2 ovf file"""

    def whd(s):
        s += "\n"
        f.write(s.encode("ASCII"))

    out = arr.astype("<f4").tobytes()
    xnodes, ynodes, znodes = arr.shape[2], arr.shape[1], arr.shape[0]
    xmin, ymin, zmin = 0, 0, 0
    xmax, ymax, zmax = xnodes * dx, ynodes * dy, znodes * dz
    xbase, ybase, _ = dx / 2, dy / 2, dz / 2
    valuedim = arr.shape[-1]
    valuelabels = "x y z"
    valueunits = "1 1 1"
    total_sim_time = "0"
    name = path.split("/")[-1]
    with open(path, "wb") as f:
        whd("# OOMMF OVF 2.0")
        whd("# Segment count: 1")
        whd("# Begin: Segment")
        whd("# Begin: Header")
        whd(f"# Title: {name}")
        whd("# meshtype: rectangular")
        whd("# meshunit: m")
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
        whd(f"# xstepsize: {dx}")
        whd(f"# ystepsize: {dy}")
        whd(f"# zstepsize: {dz}")
        whd("# End: Header")
        whd("# Begin: Data Binary 4")
        f.write(struct.pack("<f", 1234567.0))
        f.write(out)
        whd("# End: Data Binary 4")
        whd("# End: Segment")


def get_ovf_parms(ovf_path: str) -> dict:
    """Return a tuple of the shape of the ovf file at the ovf_path"""
    with open(ovf_path, "rb") as f:
        while True:
            line = f.readline().strip().decode("ASCII")
            if "valuedim" in line:
                c = int(line.split(" ")[-1])
            if "xnodes" in line:
                x = int(line.split(" ")[-1])
            if "ynodes" in line:
                y = int(line.split(" ")[-1])
            if "znodes" in line:
                z = int(line.split(" ")[-1])
            if "xstepsize" in line:
                dx = float(line.split(" ")[-1])
            if "ystepsize" in line:
                dy = float(line.split(" ")[-1])
            if "zstepsize" in line:
                dz = float(line.split(" ")[-1])
            if "Desc: Total simulation time:" in line:
                t = float(line.split("  ")[-2])
            if "End: Header" in line:
                break
    parms = {"shape": (z, y, x, c), "dx": dx, "dy": dy, "dz": dz, "t": t}
    return parms


def load_ovf(ovf_path: str) -> np.ndarray:
    """Returns an np.ndarray from the ovf"""
    with open(ovf_path, "rb") as f:
        ovf_shape = [0, 0, 0, 0]
        for _ in range(28):
            line = f.readline().strip().decode("ASCII")
            if "valuedim" in line:
                ovf_shape[3] = int(line.split(" ")[-1])
            if "xnodes" in line:
                ovf_shape[2] = int(line.split(" ")[-1])
            if "ynodes" in line:
                ovf_shape[1] = int(line.split(" ")[-1])
            if "znodes" in line:
                ovf_shape[0] = int(line.split(" ")[-1])
        count = ovf_shape[0] * ovf_shape[1] * ovf_shape[2] * ovf_shape[3] + 1
        arr = np.fromfile(f, "<f4", count=count)[1:].reshape(ovf_shape)
    return arr
