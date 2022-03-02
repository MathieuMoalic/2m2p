import os
import glob
import multiprocessing as mp
import re

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import matplotlib as mpl
import zarr
import h5py
from numcodecs import Blosc


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
        parms: dict(int, int, int, int, float, float, float) = {}
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
