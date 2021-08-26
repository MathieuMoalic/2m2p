from typing import Tuple, Optional
import os
import re
import glob
import shutil

import numpy as np

from ._utils import get_config
from ._ovf import get_ovf_parms


class Make:
    def __init__(self, llyr, load_path, tmax=None, override=False, delete_out=True):
        self.llyr = llyr
        self.ts = 0
        llyr.create_h5(override)
        out_path, mx3_path = self._get_paths(load_path)

        self.add_mx3(mx3_path)
        dset_prefixes = self._get_dset_prefixes(out_path)
        for prefix, name in dset_prefixes.items():
            self.make_dset(out_path, prefix, name=name, tmax=tmax)
        self.add_table(f"{out_path}/table.txt")
        if delete_out:
            shutil.rmtree(out_path)

    def _get_paths(self, load_path: str) -> Tuple[str, str]:
        """Cleans the input string and return the path for .out folder and .mx3 file"""
        load_path = load_path.replace(".mx3", "").replace(".out", "").replace(".h5", "")
        out_path = f"{load_path}.out"
        mx3_path = f"{load_path}.mx3"
        if not os.path.exists(out_path):
            raise NameError(f"{out_path} not found")
        if not os.path.exists(mx3_path):
            raise NameError(f"{mx3_path} not found")
        return out_path, mx3_path

    def add_mx3(self, mx3_path: str) -> None:
        """Adds the mx3 file to the f.attrs"""
        if os.path.isfile(mx3_path):
            with open(mx3_path, "r") as mx3:
                self.llyr.add_attr("mx3", mx3.read())
        else:
            print(f"{mx3_path} not found")

    def add_table(self, table_path: str, dset_name: str = "table") -> None:
        """Adds a the mumax table.txt file as a dataset"""
        if os.path.isfile(table_path):
            with open(table_path, "r") as table:
                header = table.readline()
                data = np.loadtxt(table).T
                times = data[0]
                dt = (times[-1] - times[0]) / len(times)
                print(dt)

            self.llyr.add_dset(data, dset_name)
            self.llyr.add_attr("header", header, dset_name)
            self.llyr.add_attr("dt", dt)

    def _get_dset_prefixes(self, out_path: str) -> dict:
        """From the .out folder, get the list of prefixes, each will correspond to a different dataset"""
        paths = glob.glob(f"{out_path}/*.ovf")
        prefixes = list(
            {re.sub(r"_?[\d.]*.ovf", "", path.split("/")[-1]) for path in paths}
        )
        names = {}
        prefix_to_name = get_config()
        for prefix in prefixes:
            names[prefix] = prefix
            for pattern, name in prefix_to_name.items():
                if re.findall(pattern, prefix.lower()):
                    names[prefix] = name
                    break
        return names

    def make_dset(
        self,
        out_path: str,
        prefix: str,
        name: str,
        tmax: Optional[int] = None,
    ) -> None:
        """Creates a dataset from an input .out folder path and a prefix (i.e. "m00")"""
        ovf_paths = sorted(glob.glob(f"{out_path}/{prefix}*.ovf"))[:tmax]
        # this is to calculate dt
        if self.ts < len(ovf_paths):
            self.ts = len(ovf_paths)
        # load one file to initialize the h5 dataset with the correct shape
        ovf_parms = get_ovf_parms(ovf_paths[0])
        for key in ["dx", "dy", "dz"]:
            if key not in self.llyr.attrs:
                self.llyr.add_attr(key, ovf_parms[key])

        dset_shape = (len(ovf_paths),) + ovf_parms["shape"]
        # name = self._get_dset_prefixes(prefix)
        self.llyr.load_dset(name, dset_shape, ovf_paths)
