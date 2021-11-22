import os
import configparser
import glob
import multiprocessing as mp

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from appdirs import user_config_dir
from tqdm import tqdm


def get_config() -> configparser.SectionProxy:
    config_dir = user_config_dir("llyr")
    if os.path.isfile("llyr.ini"):
        config_path = "llyr.ini"
    elif os.path.isfile(f"{config_dir}/llyr.ini"):
        config_path = f"{config_dir}/llyr.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    return config["llyr"]


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


# def load_dset(self, name: str, dset_shape: tuple, ovf_paths: list) -> None:
#     dset = llyr.create_dataset(name, dset_shape, np.float32)
#     with mp.Pool(processes=int(mp.cpu_count() - 1)) as p:
#         for i, data in enumerate(
#             tqdm(
#                 p.imap(load_ovf, ovf_paths),
#                 leave=False,
#                 desc=name,
#                 total=len(ovf_paths),
#             )
#         ):
#             dset[i] = data


# def get_ovf_parms(ovf_path: str) -> dict:
#     """Return a tuple of the shape of the ovf file at the ovf_path"""
#     with open(ovf_path, "rb") as f:
#         while True:
#             line = f.readline().strip().decode("ASCII")
#             if "valuedim" in line:
#                 c = int(line.split(" ")[-1])
#             if "xnodes" in line:
#                 x = int(line.split(" ")[-1])
#             if "ynodes" in line:
#                 y = int(line.split(" ")[-1])
#             if "znodes" in line:
#                 z = int(line.split(" ")[-1])
#             if "xstepsize" in line:
#                 dx = float(line.split(" ")[-1])
#             if "ystepsize" in line:
#                 dy = float(line.split(" ")[-1])
#             if "zstepsize" in line:
#                 dz = float(line.split(" ")[-1])
#             if "Desc: Total simulation time:" in line:
#                 t = float(line.split("  ")[-2])
#             if "End: Header" in line:
#                 break
#     parms = {"shape": (z, y, x, c), "dx": dx, "dy": dy, "dz": dz, "t": t}
#     return parms


# def load_ovf(ovf_path: str) -> np.ndarray:
#     """Returns an np.ndarray from the ovf"""
#     with open(ovf_path, "rb") as f:
#         ovf_shape = [0, 0, 0, 0]
#         for _ in range(28):
#             line = f.readline().strip().decode("ASCII")
#             if "valuedim" in line:
#                 ovf_shape[3] = int(line.split(" ")[-1])
#             if "xnodes" in line:
#                 ovf_shape[2] = int(line.split(" ")[-1])
#             if "ynodes" in line:
#                 ovf_shape[1] = int(line.split(" ")[-1])
#             if "znodes" in line:
#                 ovf_shape[0] = int(line.split(" ")[-1])
#         count = ovf_shape[0] * ovf_shape[1] * ovf_shape[2] * ovf_shape[3] + 1
#         arr = np.fromfile(f, "<f4", count=count)[1:].reshape(ovf_shape)
#     return arr
