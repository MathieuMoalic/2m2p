import os
import glob
import multiprocessing as mp
import re
import colorsys

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import matplotlib as mpl
import zarr
import h5py
from numcodecs import Blosc
import scipy as sp
import IPython


def fix_bg():
    IPython.get_ipython().run_cell_magic(
        "html",
        "",
        "<style> .cell-output-ipywidget-background {background-color: transparent !important;}</style>",
    )


def normalize(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    with np.errstate(divide="ignore", invalid="ignore"):
        out: npt.NDArray[np.float32] = arr / np.linalg.norm(arr, axis=-1)[..., None]
        return out


def rgb_int_from_vectors(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    h = np.angle(x + 1j * y, deg=True)
    s = np.sqrt(x**2 + y**2 + z**2)
    l = (z + 1) / 2  # disable: flake8
    rgb = np.zeros_like(arr, dtype=np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, n in enumerate([0, 8, 4]):
            k = (n + h / 30) % 12
            a = s * np.minimum(l, 1 - l)
            k = np.clip(np.minimum(k - 3, 9 - k), -1, 1)
            rgb[..., i] = (l - a * k) * 255
    return (rgb[..., 0] << 16) + (rgb[..., 1] << 8) + rgb[..., 2]


def hsl2rgb(hsl):
    h = hsl[..., 0] * 360
    s = hsl[..., 1]
    l = hsl[..., 2]

    rgb = np.zeros_like(hsl)
    for i, n in enumerate([0, 8, 4]):
        k = (n + h / 30) % 12
        a = s * np.minimum(l, 1 - l)
        k = np.minimum(k - 3, 9 - k)
        k = np.clip(k, -1, 1)
        rgb[..., i] = l - a * k
    rgb = np.clip(rgb, 0, 1)
    return rgb


def hsl2rgb2(hsl):
    h = hsl[..., 0] * 360
    s = hsl[..., 1]
    l = hsl[..., 2]

    rgb = np.zeros_like(hsl)
    for i, n in enumerate([0, 8, 4]):
        k = (n + h / 30) % 12
        a = s * np.minimum(l, 1 - l)
        k = np.minimum(k - 3, 9 - k)
        k = np.clip(k, -1, 1)
        rgb[..., i] = l - a * k
    rgb = np.clip(rgb, 0, 1)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            for k in range(rgb.shape[2]):
                if all(rgb[i, j, k] == 0):
                    rgb[i, j, k] = np.array([1, 1, 1])
    return rgb


def clean_glob_names(ps):
    ps = sorted([x.split("/")[-1].split(".")[0] for x in ps])
    pre_sub = ""
    for i in range(20):
        q = [pre_sub + ps[0][i] == p[: i + 1] for p in ps]
        if not all(q):
            break
        pre_sub += ps[0][i]
    post_sub = ""
    for i in range(-1, -20, -1):
        q = [ps[0][i] + post_sub == p[i:] for p in ps]
        if not all(q):
            break
        post_sub = ps[0][i] + post_sub
    ps = [p.replace(pre_sub, "").replace(post_sub, "") for p in ps]
    return pre_sub, post_sub, ps


def cspectra_b(Llyr):
    def cspectra(ps, norm=None):
        cmaps = []
        for a, b, c in zip((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            N = 256
            vals = np.ones((N, 4))
            vals[:, 0] = np.linspace(1, a, N)
            vals[:, 1] = np.linspace(1, b, N)
            vals[:, 2] = np.linspace(1, c, N)
            vals[:, 3] = np.linspace(0, 1, N)
            cmaps.append(mpl.colors.ListedColormap(vals))
        paths = glob.glob(f"{ps}/*.zarr")[:17]
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
        for c, cmap in zip([0, 1], [cmaps[0], cmaps[2]]):
            names = []
            arr = []
            for p in paths:
                m = Llyr(p)
                names.append(m.name)
                x, y = m.fft_tb(c, tmax=None, normalize=True)
                x, y = x[5:], y[5:]
                arr.append(y)
            arr = np.array(arr).T
            # norm=mpl.colors.SymLogNorm(linthresh=0.2)

            ax.imshow(
                arr,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=[0, arr.shape[1] * 2, x.min(), x.max()],
                cmap=cmap,
                norm=norm,
            )
        ax.legend(
            handles=[
                mpl.patches.Patch(color="red", label="mx"),
                mpl.patches.Patch(color="blue", label="mz"),
            ],
            fontsize=5,
        )
        _, _, ps = clean_glob_names(paths)
        ax.set_ylim(0, 15)
        ax.set_xticks(np.arange(1, arr.shape[1] * 2, 2))
        ax.set_xticklabels(ps)
        ax.set_xlabel("Ring Width (nm)")
        ax.set_ylabel("Frequency (GHz)")
        fig.tight_layout(h_pad=0.4, w_pad=0.2)
        return fig, ax

    return cspectra


def merge_table(m):
    for d in ["m", "B_ext"]:
        if f"table/{d}x" in m:
            x = m[f"table/{d}x"]
            y = m[f"table/{d}y"]
            z = m[f"table/{d}z"]
            m.create_dataset(f"table/{d}", data=np.array([x, y, z]).T)
            del m[f"table/{d}x"]
            del m[f"table/{d}y"]
            del m[f"table/{d}z"]


def h5_to_zarr(p, remove=False):
    source = h5py.File(p, "r")
    dest = zarr.open(p.replace(".h5", ".zarr"), mode="a")
    print("Copying:", p)
    zarr.copy_all(source, dest)
    print("Merging tables ..")
    merge_table(dest)
    source.close()
    print("Removing ...")
    if remove:
        os.remove(p)
    print("Done")


def load_ovf(path: str):
    with open(path, "rb") as f:
        dims = np.array([0, 0, 0, 0])
        while True:
            line = f.readline().strip().decode("ASCII")
            if "valuedim" in line:
                dims[3] = int(line.split(" ")[-1])
            if "xnodes" in line:
                dims[2] = int(line.split(" ")[-1])
            if "ynodes" in line:
                dims[1] = int(line.split(" ")[-1])
            if "znodes" in line:
                dims[0] = int(line.split(" ")[-1])
            if "Begin: Data" in line:
                break
        count = int(dims[0] * dims[1] * dims[2] * dims[3] + 1)
        arr = np.fromfile(f, "<f4", count=count)[1:].reshape(dims)
    return arr


def get_ovf_parms(path: str):
    with open(path, "rb") as f:
        parms = {}
        while True:
            line = f.readline().strip().decode("ASCII")
            if "valuedim" in line:
                parms["comp"] = int(line.split(" ")[-1])
            if "xnodes" in line:
                parms["Nx"] = int(line.split(" ")[-1])
            if "ynodes" in line:
                parms["Ny"] = int(line.split(" ")[-1])
            if "znodes" in line:
                parms["Nz"] = int(line.split(" ")[-1])
            if "xstepsize" in line:
                parms["dx"] = float(line.split(" ")[-1])
            if "ystepsize" in line:
                parms["dy"] = float(line.split(" ")[-1])
            if "zstepsize" in line:
                parms["dz"] = float(line.split(" ")[-1])
            if "Begin: Data" in line:
                break
    return parms


def out_to_zarr(out_path: str, zarr_path: str, tmax=None):
    r = re.compile(r"(.*)(\d{6})")
    ovfs = sorted(glob.glob(f"{out_path}/*.ovf"))
    ovfs = [p.split("/")[-1].replace(".ovf", "") for p in ovfs]
    dsets = []
    for ovf in ovfs:
        m = r.match(ovf)
        if m:
            dset = m.groups()[0]
        else:
            dset = ovf
        if dset not in dsets:
            dsets.append(dset)
    m = zarr.open(zarr_path)
    for dset in dsets:
        ovfs = sorted(glob.glob(f"{out_path}/{dset}*.ovf"))[:tmax]
        parms = get_ovf_parms(ovfs[0])
        dset_shape = (len(ovfs), parms["Nz"], parms["Ny"], parms["Nx"], parms["comp"])
        zarr_dset = m.create_dataset(
            dset,
            shape=dset_shape,
            chunks=(5, parms["Nz"], 64, 64, parms["comp"]),
            dtype=np.float32,
            compressor=Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE),
            overwrite=True,
        )
        pool = mp.Pool(processes=int(mp.cpu_count() - 1))
        for i, d in enumerate(pool.imap(load_ovf, ovfs)):
            zarr_dset[i] = d


def get_b(x):
    return float(x.split("_")[-1].replace(".ovf", ""))


def out_to_zarr2(path: str):
    m = zarr.open(f"{path}.zarr")
    ovfs = sorted(glob.glob(f"{path}/m*.ovf"), key=get_b, reverse=False)
    parms = get_ovf_parms(ovfs[0])
    dset_shape = (len(ovfs), parms["Nz"], parms["Ny"], parms["Nx"], parms["comp"])
    zarr_dset = m.create_dataset(
        "m_down",
        shape=dset_shape,
        chunks=(5, parms["Nz"], 64, 64, parms["comp"]),
        dtype=np.float32,
        compressor=Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE),
        overwrite=True,
    )
    pool = mp.Pool(processes=int(mp.cpu_count() - 1))
    for i, d in enumerate(pool.imap(load_ovf, ovfs)):
        zarr_dset[i] = d
    zarr_dset.attrs["B_ext"] = [get_b(ovf) for ovf in ovfs]


def rechunk():
    import rechunker
    import zarr
    import shutil

    # paths = glob("./ADL_paper/*/*.zarr")
    paths = ["./ADL_paper/ref.zarr"]
    intermediate = "/tmp/intermediate.zarr"
    for p in paths:
        print(p)
        source = zarr.open(p).m
        # source = zarr.open(p).modes.m.arr
        if source.shape == (501, 1, 512, 512, 3):
            # if source.shape == (251, 1, 512, 512, 3):
            target = f"{p}/m3"
            # target = f"{p}/modes/m/arr2"
            rechunked = rechunker.rechunk(
                source,
                target_chunks=(1, 1, 64, 64, 3),
                max_mem="40GB",
                target_store=target,
                temp_store=intermediate,
            )
            rechunked.execute()
            # shutil.rmtree(f"{p}/modes/m/arr")
            # zarr.open(p).move("m2","m")
            # zarr.open(p).move("modes/m/arr2","modes/m/arr")
        else:
            print("Wrong shape ?")


def get_cmaps():
    cmaps = []
    for a, b, c in zip((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        N = 256
        vals = np.ones((N, 4))
        vals[:, 0] = np.linspace(1, a, N)
        vals[:, 1] = np.linspace(1, b, N)
        vals[:, 2] = np.linspace(1, c, N)
        vals[:, 3] = np.linspace(0, 1, N)
        cmaps.append(mpl.colors.ListedColormap(vals))
    handles = [
        mpl.patches.Patch(color="red", label="mx"),
        mpl.patches.Patch(color="green", label="my"),
        mpl.patches.Patch(color="blue", label="mz"),
    ]
    return cmaps, handles


class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=0.0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


import struct


def save_ovf(path: str, arr: np.ndarray, dx: float, dy: float, dz: float) -> None:
    """Saves the given dataset for a given t to a valid OOMMF V2 ovf file"""

    def whd(s):
        s += "\n"
        f.write(s.encode("ASCII"))

    out = arr.astype("<f4")
    out = out.tobytes()

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


def trans_ax_to_data(ax, rec):
    x0, y0, width, height = rec
    xmin, xmax = ax.get_xlim()
    x = np.abs(xmax) + np.abs(xmin)
    ymin, ymax = ax.get_ylim()
    y = np.abs(ymax) + np.abs(ymin)
    new_rec = [x0 * x, y0 * y, width * x, height * y]
    print(new_rec)
    return new_rec


def add_radial_phase_colormap(ax, rec=None):
    legend = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "color-legend.png"
    )
    if rec is None:
        rec = [0.03, 0.03, 0.25, 0.25]
    cax = ax.inset_axes(rec)
    cax.axis("off")
    im = mpl.image.imread(legend)
    cax.imshow(im, origin="upper")


def add_radial_phase_colormap2(ax, rec=None):
    def func1(hsl):
        return np.array(colorsys.hls_to_rgb(hsl[0] / (2 * np.pi), hsl[1], hsl[2]))

    if rec is None:
        rec = [0.85, 0.85, 0.15, 0.15]
    cax = plt.axes(rec, polar=True)
    p1, p2 = ax.get_position(), cax.get_position()
    cax.set_position([p1.x1 - p2.width, p1.y1 - p2.height, p2.width, p2.height])

    theta = np.linspace(0, 2 * np.pi, 360)
    r = np.arange(0, 100, 1)
    hls = np.ones((theta.size * r.size, 3))

    hls[:, 0] = np.tile(theta, r.size)
    white_pad = 20
    black_pad = 10
    hls[: white_pad * theta.size, 1] = 1
    hls[-black_pad * theta.size :, 1] = 0
    hls[white_pad * theta.size : -black_pad * theta.size, 1] = np.repeat(
        np.linspace(1, 0, r.size - white_pad - black_pad), theta.size
    )
    rgb = np.apply_along_axis(func1, 1, hls)
    cax.pcolormesh(
        theta,
        r,
        np.zeros((r.size, theta.size)),
        color=rgb,
        shading="nearest",
    )
    cax.spines["polar"].set_visible(False)
    cax.set(yticks=[], xticks=[])
    # up_symbol = dict(x=0.5, y=0.5, name=r"$\bigodot$")
    # down_symbol = dict(x=0.1, y=0.5, name=r"$\bigotimes$")
    # for s in [up_symbol, down_symbol]:
    #     cax.text(
    #         s["x"],
    #         s["y"],
    #         s["name"],
    #         color="#3b5bff",
    #         horizontalalignment="center",
    #         verticalalignment="center",
    #         fontsize=5,
    #         transform=cax.transAxes,
    #     )


# def get_azymutal_number(path, r):
#     fig, axes = plt.subplots(1, 3, figsize=(14, 5))
#     for c in range(3):
#         arr = op(path).get_mode("m", 6.66, c)[0]
#         x, y = np.arange(arr.shape[0]), np.arange(arr.shape[0])
#         vinterp = np.vectorize(sp.interpolate.interp2d(x, y, np.real(arr)))
#         xcenter = len(x) / 2
#         ycenter = len(y) / 2
#         arclen = 2 * np.pi * r
#         angle = np.linspace(0, 2 * np.pi, int(arclen * 2), endpoint=False)
#         value = vinterp(xcenter + r * np.sin(angle), ycenter + r * np.cos(angle))
#         axes[c].plot(angle, value)
