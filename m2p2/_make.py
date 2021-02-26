import numpy as np
from glob import glob
import os
import re
import h5py  # type: ignore
import multiprocessing as mp
from typing import *


class Make:
    path: str
    name: str

    def load_ovf(self, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            dims: np.ndarray = np.array([0, 0, 0, 0])
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
            count: int = int(dims[0] * dims[1] * dims[2] * dims[3] + 1)
            arr: np.ndarray = np.fromfile(f, "<f4", count=count)[1:].reshape(dims)
        return arr

    def parse_script(self) -> None:
        with h5py.File(self.path, "a") as f:
            line: str
            key: str
            value: float
            key_match: Optional[Match[str]]
            value_match: Optional[Match[str]]
            region_match: Optional[Match[str]]

            def get_attribute(key_re: str, value_re: str) -> None:
                key_match = re.search(key_re, line)
                value_match = re.search(value_re, line)
                if isinstance(key_match, str) and isinstance(value_match, str):
                    key = key_match.group()
                    value = float(value_match.group())
                    f.attrs[key] = value

            for line in f.attrs["script"].split("\n"):
                line = line.lower()

                # matches magnetic parameters
                if "setregion" in line and "anisu" not in line:
                    region_match = re.search(r"\w+_region", line)
                    key_match = re.search(r"\w+", line)
                    value_match = re.search(r"\d+\.?\d*e?-?\d*", line[4:])

                    if (
                        isinstance(key_match, str)
                        and isinstance(value_match, str)
                        and isinstance(region_match, str)
                    ):
                        key = (
                            region_match.group().split("_")[0] + "_" + key_match.group()
                        )
                        value = float(value_match.group())
                        f.attrs[key] = value
                        continue

                # matches cell dimensions
                get_attribute(r"[d][xyz]", r"[\d.e-]+")
                # matches amplitude
                get_attribute(r"amps +:=", r"[\d.e-]+")
                # matches frequency
                get_attribute(r"f +:=", r"[\d.e-]+")

    def make(self, loadpath: str) -> None:
        if os.path.isfile(self.path):
            input_string: str = input(
                f"{self.path} already exists, overwrite it [y/n]?"
            )
            if input_string.lower() in ["y", "yes"]:
                with h5py.File(self.path, "w") as f:
                    pass
            else:
                return

        with h5py.File(self.path, "a") as f:
            # save script
            with open(f"{loadpath}.mx3", "r") as script:
                f.attrs["script"] = script.read()

            # save table.txt
            if os.path.isfile(f"{loadpath}.out/table.txt"):
                with open(f"{loadpath}.out/table.txt", "r") as table:
                    header: str = table.readline()
                    tab: np.ndarray = np.loadtxt(table).T
                    tableds = f.create_dataset("table", data=tab)
                    tableds.attrs["header"] = header
                    dt: float = (tab[0, -1] - tab[0, 0]) / (tab.shape[1] - 1)
                    f.attrs["dt"] = dt

            # get names of datasets
            heads: List[str] = list(
                set([os.path.basename(x)[:-8] for x in glob(f"{loadpath}.out/*.ovf")])
            )
            dset_names: List[str] = []
            for i, head in enumerate(heads):
                if "zrange4" in head:
                    dset_names.append("ND")
                elif "zrange2" in head:
                    dset_names.append("WG")
                elif "st" in head:
                    dset_names.append("stable")
                else:
                    input_name = input(f"{head} :")
                    if input_name == "del":
                        del heads[i]
                    else:
                        dset_names.append(input_name)
            print(f"{self.name} is saving the datasets :", end=" ")
            for dset_name in dset_names:
                print(dset_name, end=" ")
            print("")

            # get list of ovf files and shapes
            ovf_list: List[List[str]] = []
            ovf_shape: List[Tuple[Union[int, slice], ...]] = []
            for head in heads:
                L: List[str] = sorted(glob(f"{loadpath}.out/{head}*.ovf"))
                ovf_list.append(L)
                shape: Tuple[Union[int, slice], ...] = self.load_ovf(L[0]).shape
                ovf_shape.append((len(L),) + shape)

            # save datasets
            for ovfs, shape, dset_name in zip(ovf_list, ovf_shape, dset_names):
                dset = f.create_dataset(dset_name, shape, np.float32)
                pool = mp.Pool(processes=int(mp.cpu_count() - 1))
                for i, d in enumerate(pool.imap(self.load_ovf, ovfs)):
                    dset[i] = d
                pool.close()
                pool.join()

            self.parse_script()
